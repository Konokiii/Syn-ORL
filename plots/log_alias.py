# 2023/11/7: Baseline IQL and synthetic IQL over 10 seeds
iql_baseline = 'iql_preMnone'
iql_mdp_qv = 'iql_preMmdp_fd_QV_ns100_pt1'
iql_mdp_q = 'iql_preMmdp_fd_onlyQ_ns100_pt1'
iql_mdp_v = 'iql_preMmdp_fd_onlyV_ns100_pt1'

# 2023/11/8: Add 7 extra seeds to baseline IQL/IQL_MDP; Add IQL-1 variants with Ant env
iql1_baseline = 'iql-1_preMnone'
iql1_mdp_qv = 'iql-1_preMmdp_fd_QV_ns100_pt1'
iql1_mdp_q = 'iql-1_preMmdp_fd_onlyQ_ns100_pt1'
iql1_mdp_v = 'iql-1_preMmdp_fd_onlyV_ns100_pt1'

# 2023/11/9: Remove cosine lr scheduler
iql1_nocos_baseline = 'iql-1_cosLRFalse_preMnone'
iql1_nocos_mdp_qv = 'iql-1_cosLRFalse_preMmdp_fd_QV_ns100_pt1'
iql1_nocos_mdp_q = 'iql-1_cosLRFalse_preMmdp_fd_onlyQ_ns100_pt1'
iql1_nocos_mdp_v = 'iql-1_cosLRFalse_preMmdp_fd_onlyV_ns100_pt1'
