import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/pat/irs-workspace/src/IRS_2025_group3/install/hand_solo_arm'
