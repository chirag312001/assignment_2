#!/usr/bin/env python3
import os
import argparse
import m5
from pathlib import Path
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.ruby.mesi_two_level_cache_hierarchy import MESITwoLevelCacheHierarchy
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.isas import ISA
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from m5.objects import DerivO3CPU, BiModeBP, GshareBP, LocalBP, TournamentBP
# defining dictionary for branch predictor
bp_dict = {
    "local": lambda: LocalBP(),
    "tournament": lambda: TournamentBP(choicePredictorSize=4096),
    "gshare": lambda: GshareBP(globalPredictorSize=12, PHTPredictorSize=8192),
    "bimode": lambda: BiModeBP(globalPredictorSize=4096, globalCtrBits=2),
}

class MyOutOfOrderCore(BaseCPUCore):
    def __init__(self, width=8, rob_size=192, num_int_regs=256, num_fp_regs=256):
        super().__init__(DerivO3CPU(), ISA.X86)
        self.core.fetchWidth = width
        self.core.decodeWidth = width
        self.core.renameWidth = width
        self.core.issueWidth = width
        self.core.wbWidth = width
        self.core.commitWidth = width

        self.core.numROBEntries = rob_size
        self.core.numPhysIntRegs = num_int_regs
        self.core.numPhysFloatRegs = num_fp_regs
        

        # Default branch predictor (you can change via code below)
        self.core.branchPred = LocalBP()
        # Examples (uncomment one to use):
        # self.core.branchPred = TournamentBP(choicePredictorSize = 4096)
        # self.core.branchPred = GshareBP(globalPredictorSize = 12, PHTPredictorSize = 8192)
        # self.core.branchPred = BiModeBP(globalPredictorSize=4096, globalCtrBits=2)

        self.core.LQEntries = 128
        self.core.SQEntries = 128

class MyOutOfOrderProcessor(BaseCPUProcessor):
    def __init__(self, width=8, rob_size=192, num_int_regs=256, num_fp_regs=256):
        cores = [MyOutOfOrderCore(width, rob_size, num_int_regs, num_fp_regs)]
        super().__init__(cores)



# passing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cmd")
parser.add_argument("--options", nargs="*")
parser.add_argument("--bp")
parser.add_argument("--rob", type= int)
parser.add_argument("--iq", type= int)



# args = parser.parse_args()
args = parser.parse_args()
binary_path = args.cmd
binary_args = args.options
bp_key = args.bp
rob_value = args.rob
iq_entry  = args.iq


# printing
print(f"[CONFIG] Binary : {binary_path}")
print(f"[CONFIG] Arguments : {binary_args}")
print(f"[CONFIG] Arguments : {bp_key}")
print(f"[CONFIG] Arguments : {rob_value}")
print(f"[CONFIG] Arguments : {iq_entry}")



# changing processor atttributes
processor = MyOutOfOrderProcessor(width=8, rob_size=192, num_int_regs=256, num_fp_regs=256)

bp_constructor = bp_dict.get(bp_key)
processor.cores[0].core.branchPred = bp_constructor()
processor.cores[0].core.numROBEntries = rob_value
processor.cores[0].core.fetchWidth = 4
processor.cores[0].core.issueWidth = 4
processor.cores[0].core.numIQEntries = iq_entry
processor.cores[0].core.max_insts_any_thread = 100_000_000


memory = SingleChannelDDR4_2400(size="2GB")
cache_hierarchy = MESITwoLevelCacheHierarchy(
    l1d_size="16kB", l1d_assoc=8,
    l1i_size="16kB", l1i_assoc=8,
    l2_size="256kB", l2_assoc=16,
    num_l2_banks=1,
)

board = SimpleBoard(
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
    clk_freq="3GHz",
)

if binary_args == "s":
    board.set_se_binary_workload(BinaryResource(binary_path))
else:
    board.set_se_binary_workload(BinaryResource(binary_path), arguments=binary_args)



sim = Simulator(board=board)
sim.run()
