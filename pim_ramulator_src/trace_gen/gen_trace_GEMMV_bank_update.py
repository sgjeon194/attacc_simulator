import argparse
import math
import copy
import numpy as np

model = "gpt-3-175B"

k = 12288
n = 12288

max_L = 2048
data_size = 16  # FP 16

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32  # byte
n_mac = 16


# Granularity size
HBM_GS = {}
HBM_GS["col"] = prefetch_size
HBM_GS["row"] = n_col * HBM_GS["col"]
HBM_GS["ba"] = n_row * HBM_GS["row"]
HBM_GS["bg"] = n_bank * HBM_GS["ba"]
HBM_GS["rank"] = n_bg * HBM_GS["bg"]
HBM_GS["pch"] = n_rank * HBM_GS["rank"]
HBM_GS["ch"] = n_pch * HBM_GS["pch"]
HBM_GS["hbm"] = n_channel * HBM_GS["ch"]
HBM_GS["attacc"] = max_n_hbm * HBM_GS["hbm"]


## --------------------------------------  HBM memory space -----------------------------------------##
## ------|  legacy CH  |  pCH  |  rank  | BG | BA |  row index  |  column index  |  access granularity  |------ ##
## bits  |     4       |   1   |   1   | 2  | 2  |     14      |        5       |          5           |       ##

## ----------------------------  Commands -------------------------------##
## MACAB: 8tCK (tCCDLx 2)
##  WRGB: 4tCK (write to SRAM not DRAM)
##  MVSB: 4tCK
##  MVGB: 4tCK
##  SFM: 16tCK (for L = 256)

cmd_score_wrgb = []
cmd_score_mac = []

valid_channels = []


def cmd_list_reset():
    cmd_score_wrgb = []
    cmd_score_mac = []

    valid_channel = []


def gemv(partial_n, weight_addr, num_itr, valid_channel=n_channel):
    cmd_score_wrgb.append([])
    cmd_score_mac.append([])

    valid_channels.append(valid_channel)

    def score_cpvec(addr_offset):
        ## (pCH) C, C, R, R (MAC)
        ## write input vector to gemv buffer
        # number of partition = (R parallel units)

        # Data broadcasting for pch, rank, bg, and ba
        for ba_idx in range(n_bank):  # number of partitions
            for col_idx in range(math.ceil(k / n_bank / n_mac)):
                for lch in range(math.ceil(valid_channel)):
                    # GEMV buffer address, col granularity = 1
                    addr = (
                        addr_offset
                        + lch * HBM_GS["ch"]
                        + ba_idx * HBM_GS["ba"]
                        + col_idx
                    )
                    hex_addr = hex(addr)[2:]
                    cmd_score_wrgb[-1].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

    def score_mac(addr_offset, n):
        ## (pCH) C, C, R, R (MAC)
        # MAC and move output vector to softmax buffer
        ## Vector (1 x k) x Matrix (k x n) multiplication
        ## GEMV unit = adder tree mode
        for n_idx in range(math.ceil(n / n_pch / n_rank / n_bg)):  # channel당 bank group 수 16 각 Bank group 당 partial_n / 16 = 24 column을 담당
            cmd_score_mac[-1].append([])
            for k_idx in range(math.ceil(k / n_bank / n_mac)):  # n_bank * n_mac = 64 bank group 하나에 들어있는 총 mac unit 수 k / 64 = 192
                idx = n_idx * math.ceil(k / n_bank / n_mac) + k_idx

                # All bank command (legacy channel)
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS["ch"] + idx * HBM_GS["col"]
                    hex_addr = hex(addr)[2:]
                    cmd_score_mac[-1][-1].append(
                        "PIM_MAC_AB 0x{0:0>8}".format(hex_addr)
                    )

    score_cpvec(weight_addr)

    for itr in range(num_itr):
        score_mac(weight_addr, partial_n)


# n_head and n_req = n_req per a HBM
def run_gemm(m, k, n, trace_file_name): 
    partition_size = math.ceil(max_L * k / (n_pch * n_rank * n_bg * n_bank))
    # head_offset = partition_size
    # v_offset = pow(2, 23)

    # cmd_list_reset()
    # ##-- Generate Commands --##
    # num_itr = math.ceil(m / (n_channel))
    # for itr in range(num_itr):
    #     remainder = 0
    #     if m / ((itr + 1) * n_channel) < 1:
    #         remainder = m % n_channel
    #     key_addr = itr * partition_size
    #     val_addr = key_addr + v_offset
    #     if remainder == 0:
    #         gemv(n, key_addr, val_addr, itr)
    #     else:
    #         gemv(n, key_addr, val_addr, itr, remainder)


    max_parameter_per_bank = pow(2, 23)
    width_per_channel = math.ceil(n / n_channel) # 768
    num_itr = math.ceil(width_per_channel * k / max_parameter_per_bank) # 2
    width_per_channel /= num_itr # 384
    weight_addr = 0

    for i in range(m):
        gemv(width_per_channel, weight_addr, num_itr)            

    ##-- Ovelapping Commands --##
    barrier = []
    for lch in range(n_channel):
        addr = lch * HBM_GS['ch']
        hex_addr = hex(addr)[2:]
        barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

    total_cmd = []
    for i in range(0, m - 1, 2):
        # Head0: Score
            ## WRGB
        total_cmd += cmd_score_wrgb[i]
            ## dummy MAC
        if i == 0:
            for j in range(valid_channels[i]):
                total_cmd.append(cmd_score_mac[i][0][j])
            ## BARRIER
        total_cmd += barrier

        length = math.ceil(n/n_pch/n_rank/n_bg/16)
        for j in range(0, length+1):
            ## MAC (Head0)
            if not j == length:
                stride = 16;
                for k_idx in range(stride):
                    if (j*stride+k_idx) >= len(cmd_score_mac[i]):
                        break;
                    total_cmd += cmd_score_mac[i][j*stride+k_idx]

            ## WRGB (Head1)
            if not j == length:
                stride = int(n_bank*math.ceil(k/n_bank /n_mac)*math.ceil(valid_channels[i+1])/length);
                for k_idx in range(stride):
                    if (j*stride+k_idx) >= len(cmd_score_wrgb[i+1]):
                        break;
                    total_cmd.append(cmd_score_wrgb[i+1][j*stride + k_idx])
            ## BARRIER
            if not j == length:
                total_cmd += barrier

        # Head0: SoftMax, Head1: Score
        length = math.ceil(n/n_pch/n_rank/n_bg/16)
        for j in range(0, length+1):
            ## MAC (Head1)
            if not j == length:
                stride = 16;
                for k_idx in range(stride):
                    if (j*stride+k_idx) >= len(cmd_score_mac[i+1]):
                        break;
                    total_cmd += cmd_score_mac[i+1][j*stride+k_idx]

    if m % 2 != 0:
        i = m - 1

        # Score
            ## WRGB
        total_cmd += cmd_score_wrgb[i]
            ## BARRIER
        total_cmd += barrier

        length = math.ceil(n/n_pch/n_rank/n_bg/16)
        for j in range(0, length+1):
            ## MAC
            if not j == length:
                stride = 16;
                for k_idx in range(stride):
                    if (j*stride+k_idx) >= len(cmd_score_mac[i]):
                        break;
                    total_cmd += cmd_score_mac[i][j*stride+k_idx]
            
            ## BARRIER
            if not j == length:
                total_cmd += barrier

    trace_file = open(trace_file_name, 'w')
    for cmd in total_cmd:
        trace_file.write(cmd + "\n")

    trace_file.close()


def main():
    global m, k, n, max_L, data_size, n_mac
    
    parser = argparse.ArgumentParser(
        description="Output path and operation infos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--row", type=int, default=1, help="num of GEMVs for 1 hbm, default=1"
    )
    parser.add_argument(
        "-k", "--hiddensize", type=int, default=4096, help="layer hidden size, default=4096"
    )
    parser.add_argument(
        "-n", "--col", type=int, default=4096, help="Result col num, default=4096",
    )
    parser.add_argument(
        "-maxl", "--maxlen", type=int, default=4096, help="maximum len, default= 4096"
    )
    parser.add_argument(
        "-db", "--dbyte", type=int, default=2, help="data type (B), default= 2"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="attacc_bank.trace", help="output path"
    )

    args = parser.parse_args()

    m = args.row # n_head_per_hbm
    k = args.hiddensize # dhead
    n = args.col # L
    max_L = args.maxlen

    data_size = args.dbyte
    n_mac = int(HBM_GS["col"] / data_size)

    print("------   Make a trace of bank-level AttAcc   ------")

    args_dict = vars(args)
    print("All Arguments:")
    for key, value in args_dict.items():
        print(f"     {key}: {value}")
    print("---------------------------------------------------")
    
    run_gemm(m, k, n, args.output)


if __name__ == "__main__":
    main()
