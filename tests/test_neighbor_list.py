from pprint import pprint

import torch

from deepmc.models.components.MoCap_Solver.skeleton import SkeletonLinear, build_edge_topology, find_neighbor
from deepmc.models.components.MoCap_Solver.MC_Encoder import TS_enc


def main():
    ts_enc = TS_enc()
    pprint(ts_enc.topologies)

if __name__ == "__main__":
    main()