# ==============================================================================
# BENCHMARK: Extrapolating CPU Parallelization for fishSim vs CKMRpop vs GPU
# Demography: Conn et al. 2025 (Simulation Study 2)
# ==============================================================================

library(doParallel)
library(foreach)
library(CKMRpop)
library(fishSim)
library(dplyr)
library(reticulate)

set.seed(12345)

# ------------------------------------------------------------------------------
# 1. HELPER FUNCTION: SOLVE SURV0 FOR TARGET LAMBDA
# ------------------------------------------------------------------------------
get_surv0_for_lambda <- function(target_lambda) {
  f <- function(s0) {
    L_temp <- matrix(0,5,5)
    L_temp[1,] <- c(0,0,.9,.9,0)
    L_temp[2,1] <- s0
    L_temp[3,2] <- L_temp[4,3] <- L_temp[5,4] <- 0.9
    return(Re(eigen(L_temp)$values[1]) - target_lambda)
  }
  uniroot(f, interval=c(0, 1))$root
}

# ------------------------------------------------------------------------------
# 2. EMBEDDED PYTHON GPU SIMULATOR (Stripped for Benchmarking)
# ------------------------------------------------------------------------------
cat("Initializing Python GPU environment...\n")
py_code <- "
import torch
import time
import pandas as pd

def run_gpu_benchmark_scenario(n_sims, batch_size, n_init_val, surv0_val, device_name='cuda'):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device_name)

    B = batch_size
    n_yrs = 30
    n_ages = 5
    
    mtdna_props = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
    samples_per_yr = 25
    sample_yrs = 10
    total_sampled = samples_per_yr * sample_yrs
    
    # Calculate worst case max pop manually to ensure buffer
    L_temp = torch.zeros((5, 5), dtype=torch.float64, device='cpu')
    L_temp[0, 2] = 0.9; L_temp[0, 3] = 0.9
    L_temp[1, 0] = surv0_val
    L_temp[2, 1] = 0.9; L_temp[3, 2] = 0.9; L_temp[4, 3] = 0.9
    vals, vecs = torch.linalg.eig(L_temp)
    max_idx = vals.real.argmax()
    age_props_base = vecs[:, max_idx].real
    age_props_base = (age_props_base / age_props_base.sum()).float()
    
    total_individuals_ever = n_init_val
    N = torch.zeros(5, dtype=torch.float32)
    N[:] = n_init_val * age_props_base
    S = [surv0_val, 0.90, 0.90, 0.90, 0.0]
    for _ in range(n_yrs):
        new_N = torch.zeros(5)
        mat_f = (N[3] + N[4]) * 0.5
        new_N[0] = mat_f * 2.0 
        new_N[1] = N[0] * S[0]
        new_N[2] = N[1] * S[1]
        new_N[3] = N[2] * S[2]
        new_N[4] = N[3] * S[3]
        total_individuals_ever += new_N[0].item()
        N = new_N
    max_pop = int(total_individuals_ever * 3.0)

    start_time = time.time()
    torch.cuda.empty_cache()

    batch_n_init = torch.full((B,), n_init_val, dtype=torch.long, device=device)
    batch_surv0 = torch.full((B,), surv0_val, dtype=torch.float32, device=device)
    
    surv_probs = torch.tensor([0.0, 0.90, 0.90, 0.90, 0.0], device=device).unsqueeze(0).repeat(B, 1)
    surv_probs[:, 0] = batch_surv0

    L_batch = torch.zeros((B, 5, 5), dtype=torch.float64, device=device)
    L_batch[:, 0, 2] = 0.9
    L_batch[:, 0, 3] = 0.9
    L_batch[:, 1, 0] = batch_surv0.double()
    L_batch[:, 2, 1] = 0.9
    L_batch[:, 3, 2] = 0.9
    L_batch[:, 4, 3] = 0.9
    
    vals, vecs = torch.linalg.eig(L_batch)
    max_indices = vals.real.argmax(dim=1)
    b_idx_eig = torch.arange(B, device=device)
    dominant_vecs = vecs[b_idx_eig, :, max_indices].real
    
    age_props_batch = torch.abs(dominant_vecs)
    age_props_batch = (age_props_batch / age_props_batch.sum(dim=1, keepdim=True)).float()

    seq = torch.arange(max_pop, device=device).unsqueeze(0).expand(B, -1)
    alive = seq < batch_n_init.unsqueeze(1)
    
    ages = torch.zeros((B, max_pop), dtype=torch.int8, device=device)
    founder_ages = torch.multinomial(age_props_batch, max_pop, replacement=True)
    ages = torch.where(alive, founder_ages.to(torch.int8), ages)

    birth_years = torch.full((B, max_pop), -1, dtype=torch.int8, device=device)
    birth_years = torch.where(alive, -ages, birth_years)

    sexes = torch.zeros((B, max_pop), dtype=torch.bool, device=device)
    random_sex = torch.rand((B, max_pop), device=device) > 0.5
    sexes = torch.where(alive, random_sex, sexes)

    mtdna = torch.zeros((B, max_pop), dtype=torch.int8, device=device)
    founder_mtdna = torch.multinomial(mtdna_props, B * max_pop, replacement=True).view(B, max_pop)
    mtdna = torch.where(alive, founder_mtdna.to(torch.int8), mtdna)

    mother_id = torch.full((B, max_pop), -1, dtype=torch.int32, device=device)
    father_id = torch.full((B, max_pop), -1, dtype=torch.int32, device=device)
    death_years = torch.full((B, max_pop), -1, dtype=torch.int8, device=device)
    
    next_idx = batch_n_init.clone().to(torch.long)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_pop)

    for yr in range(n_yrs):
        yr_sim = yr + 1
        rand_surv = torch.rand((B, max_pop), device=device)
        safe_ages = torch.clamp(ages, 0, n_ages - 1).long()
        thresholds = surv_probs[batch_idx, safe_ages]
        
        died_this_yr = alive & (rand_surv > thresholds)
        death_years[died_this_yr] = yr_sim
        alive[died_this_yr] = False
        
        max_age_mask = alive & (ages >= (n_ages - 1))
        death_years[max_age_mask] = yr_sim
        alive[max_age_mask] = False

        mat_f = alive & sexes & ((ages == 2) | (ages == 3))
        mat_m = alive & ~sexes & ((ages == 2) | (ages == 3))

        total_mat_f = mat_f.sum(dim=1)
        total_mat_m = mat_m.sum(dim=1)
        
        expected_newborns = total_mat_f.float() * 2.0
        raw_newborns = torch.poisson(expected_newborns).long()
        num_newborns = torch.where((total_mat_f > 0) & (total_mat_m > 0), raw_newborns, torch.tensor(0, device=device))
        max_newborns = num_newborns.max().item()
        
        if max_newborns > 0:
            cumsum_f = mat_f.long().cumsum(dim=1)
            cumsum_m = mat_m.long().cumsum(dim=1)
            
            rand_f = torch.rand((B, max_newborns), device=device) * total_mat_f.unsqueeze(1)
            rand_f = torch.ceil(rand_f).long().clamp(min=1)
            rand_m = torch.rand((B, max_newborns), device=device) * total_mat_m.unsqueeze(1)
            rand_m = torch.ceil(rand_m).long().clamp(min=1)
            
            chosen_moms = torch.searchsorted(cumsum_f, rand_f)
            chosen_dads = torch.searchsorted(cumsum_m, rand_m)
            
            seq_new = torch.arange(max_newborns, device=device).expand(B, -1)
            valid_newborns = seq_new < num_newborns.unsqueeze(1)
            
            insert_indices = next_idx.unsqueeze(1) + seq_new
            b_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_newborns)[valid_newborns]
            flat_insert = insert_indices[valid_newborns]
            
            valid_insert = flat_insert < max_pop
            b_idx_safe = b_indices[valid_insert]
            i_idx_safe = flat_insert[valid_insert]
            
            alive[b_idx_safe, i_idx_safe] = True
            ages[b_idx_safe, i_idx_safe] = 0
            birth_years[b_idx_safe, i_idx_safe] = yr
            sexes[b_idx_safe, i_idx_safe] = torch.rand(len(b_idx_safe), device=device) > 0.5
            
            moms_for_newborns = chosen_moms[valid_newborns][valid_insert]
            dads_for_newborns = chosen_dads[valid_newborns][valid_insert]
            
            mother_id[b_idx_safe, i_idx_safe] = moms_for_newborns.to(torch.int32)
            father_id[b_idx_safe, i_idx_safe] = dads_for_newborns.to(torch.int32)
            mtdna[b_idx_safe, i_idx_safe] = mtdna[b_idx_safe, moms_for_newborns.long()]
            
            next_idx += num_newborns

        ages[alive] += 1

    samples = torch.full((B, total_sampled), -1, dtype=torch.int32, device=device)
    samp_yr_full = torch.full((B, max_pop), -1, dtype=torch.int8, device=device)
    s_col = 0
    idx_grid = torch.arange(max_pop, device=device).unsqueeze(0).expand(B, -1)
    
    for sy in range(21, 31):
        valid_dead = (death_years == sy) & (idx_grid < next_idx.unsqueeze(1))
        
        rand_weights = torch.rand((B, max_pop), device=device)
        rand_weights = torch.where(valid_dead, rand_weights, torch.full_like(rand_weights, float('-inf')))
        
        weights_sampled, sampled_idx = torch.topk(rand_weights, samples_per_yr, dim=1)
        valid_mask = weights_sampled > float('-inf')
        sampled_idx = torch.where(valid_mask, sampled_idx.to(torch.int32), torch.tensor(-1, dtype=torch.int32, device=device))
        
        samples[:, s_col:s_col+samples_per_yr] = sampled_idx
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, samples_per_yr)
        valid_b_idx = b_idx[valid_mask]
        
        valid_sampled_idx = sampled_idx[valid_mask].long()
        samp_yr_full[valid_b_idx, valid_sampled_idx] = sy
        s_col += samples_per_yr

    B_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, total_sampled)
    valid_samp = samples != -1
    samp_ids_safe = samples.clone().long()
    samp_ids_safe[samples == -1] = 0
    
    o_by = torch.gather(birth_years, 1, samp_ids_safe)
    m_ids = torch.gather(mother_id, 1, samp_ids_safe)
    f_ids = torch.gather(father_id, 1, samp_ids_safe)
    po_matrix_flat = torch.zeros(B * sample_yrs * n_yrs, dtype=torch.int32, device=device)
    
    def accumulate_po(parent_ids):
        valid_p = valid_samp & (parent_ids != -1)
        parent_ids_safe = parent_ids.clone().long()
        parent_ids_safe[parent_ids == -1] = 0
        
        p_samp_yr = torch.gather(samp_yr_full, 1, parent_ids_safe)
        valid_match = valid_p & (p_samp_yr >= 21) & (o_by >= 0) & (o_by < n_yrs)
        
        b_match = B_idx[valid_match]
        p_sy_match = (p_samp_yr[valid_match] - 21).long()
        o_by_match = o_by[valid_match].long()
        
        flat_idx = b_match * (sample_yrs * n_yrs) + p_sy_match * n_yrs + o_by_match
        po_matrix_flat.scatter_add_(0, flat_idx.long(), torch.ones_like(flat_idx, dtype=torch.int32))

    accumulate_po(m_ids)
    accumulate_po(f_ids)
    
    samp_sy_flat = torch.repeat_interleave(torch.arange(21, 31, dtype=torch.int8, device=device), samples_per_yr)
    by_i = o_by.unsqueeze(2).expand(B, total_sampled, total_sampled)
    by_j = o_by.unsqueeze(1).expand(B, total_sampled, total_sampled)
    sy_i = samp_sy_flat.unsqueeze(0).unsqueeze(2).expand(B, total_sampled, total_sampled)
    
    valid_i = valid_samp.unsqueeze(2).expand(B, total_sampled, total_sampled)
    valid_j = valid_samp.unsqueeze(1).expand(B, total_sampled, total_sampled)
    comp_mask = valid_i & valid_j & (by_j > by_i + 2) & (by_j < sy_i) & (by_j >= 0) & (by_j < n_yrs)
    
    comp_matrix_flat = torch.zeros(B * sample_yrs * n_yrs, dtype=torch.int32, device=device)
    b_idx_comp = B_idx.unsqueeze(2).expand(B, total_sampled, total_sampled)[comp_mask]
    sy_match_comp = (sy_i[comp_mask] - 21).long()
    by_match_comp = by_j[comp_mask].long()
    
    flat_idx_comp = b_idx_comp * (sample_yrs * n_yrs) + sy_match_comp * n_yrs + by_match_comp
    comp_matrix_flat.scatter_add_(0, flat_idx_comp.long(), torch.ones_like(flat_idx_comp, dtype=torch.int32))

    if device.type == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    
    sims_per_sec = B / (end_time - start_time)
    return sims_per_sec
"
py_run_string(py_code)

# ------------------------------------------------------------------------------
# 3. NATIVE FISHSIM WRAPPER (Stripped for Benchmark)
# ------------------------------------------------------------------------------
run_fishsim_rep <- function(inputs) {
  N_init = inputs$N_init
  curMort = inputs$curMort
  L = inputs$L
  Age_props = inputs$Age_props
  
  n_yrs = inputs$n_yrs
  n_ages = inputs$n_ages
  N_yrs_sample = inputs$N_yrs_sample
  n_yrs_sample = length(N_yrs_sample)
  samples_per_year = inputs$samples_per_year
  Repro = inputs$Repro
  mat_age = inputs$mat_age
  
  archive <- make_archive()
  indiv = makeFounders(pop=N_init, osr=c(0.5,0.5), stocks=1, minAge=1, maxAge=5, survCurv=Age_props)
  
  for(iyr in 1:n_yrs){
    indiv <- mort(indiv = indiv, type = "age", ageMort = c(0,curMort), year = iyr, maxAge=n_ages)
    archive <- archive_dead(indiv = indiv, archive = archive)
    indiv <- remove_dead(indiv = indiv)
    
    if(sum(indiv$Sex=="F" & indiv$AgeLast %in% c(3,4)) > 0 & sum(indiv$Sex=="M" & indiv$AgeLast %in% c(3,4)) > 0) {
      indiv <- altMate(indiv = indiv, type='ageSex', maleCurve=Repro, femaleCurve=Repro, osr=c(0.5,0.5), batchSize=2, fecundityDist="poisson", year=iyr)
      indiv <- birthdays(indiv = indiv)
    }
  }
  
  for(isampyr in 1:n_yrs_sample){
    cur_yr = N_yrs_sample[isampyr]
    Cur_which = which(archive$DeathY==cur_yr)
    if(length(Cur_which) >= samples_per_year) {
      archive[sample(Cur_which,samples_per_year),"SampY"] = N_yrs_sample[isampyr]
    }
  }
  
  indiv = rbind(archive, indiv)
  
  if(nrow(indiv) > 0) {
    Kin <- quickin(indiv, max_gen=2)  
  }
  return(TRUE)
}

# ------------------------------------------------------------------------------
# 4. NATIVE CKMRPOP WRAPPER (Dynamic Parameters)
# ------------------------------------------------------------------------------
run_ckmrpop_rep <- function(inputs) {
  spip_pars <- list(
    `max-age`          = 4,                
    `number-of-years`  = 30,               
    
    `initial-females`  = inputs$init_str,
    `initial-males`    = inputs$init_str,
    
    `fem-surv-probs`   = inputs$surv_str,  
    `male-surv-probs`  = inputs$surv_str,  
    
    `fem-prob-repro`   = "0 0 1 1",
    `male-prob-repro`  = "0 0 1 1",
    
    `fem-asrf`         = "0 0 1 1", 
    `male-asrp`        = "0 0 1 1",
    
    `cohort-size`      = inputs$cohort_str,
    
    `discard-all`        = 0,            
    `gtyp-ppn-fem-post`  = "21-30 0.14 0.14 0.14 0.14", 
    `gtyp-ppn-male-post` = "21-30 0.14 0.14 0.14 0.14"  
  )
  
  spip_dir <- run_spip(pars = spip_pars)
  spip_output <- slurp_spip(spip_dir, 2)
  kin_pairs <- compile_related_pairs(spip_output$samples)
  unlink(spip_dir, recursive = TRUE)
  
  return(TRUE)
}

# ------------------------------------------------------------------------------
# 5. MASTER EXECUTION HARNESS
# ------------------------------------------------------------------------------
# Setup parallel tracking
max_cores <- 12
test_sims_per_core <- 10
total_cpu_sims <- max_cores * test_sims_per_core

# Define scenarios
N_inits <- c(250, 500, 1000)
lambdas <- c(0.95, 1.0, 1.05)

# MODIFIED: Changed results_df to wide format with requested column names
results_df <- data.frame(
  N = numeric(),
  lambda = numeric(),
  `fishsim sims/sec` = numeric(),
  `CKMRpop sims/sec` = numeric(),
  `Python/GPU sims/sec` = numeric(),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

cat("Starting Master Benchmark Loop...\n\n")

for (N in N_inits) {
  for (lam in lambdas) {
    cat(sprintf("Evaluating Scenario -> N_init: %d | Lambda: %.2f\n", N, lam))
    
    # Solve exact required surv0
    surv0_val <- get_surv0_for_lambda(lam)
    
    # Calculate demography
    L <- matrix(0,5,5)
    L[1,] <- c(0,0,.9,.9,0)
    L[2,1] <- surv0_val
    L[3,2] <- L[4,3] <- L[5,4] <- 0.9
    Age_props <- Re(eigen(L)$vectors[,1] / sum(eigen(L)$vectors[,1]))
    curMort <- 1 - c(surv0_val, 0.9, 0.9, 0.9, 0)
    
    # Prepare CKMRpop strings
    # 4 elements expected by max-age=4. Combine ages 3 and 4 into a plus group.
    init_props <- c(Age_props[1], Age_props[2], Age_props[3], Age_props[4] + Age_props[5])
    init_counts <- round((N / 2) * init_props)
    init_str <- paste(init_counts, collapse=" ")
    surv_str <- paste(round(surv0_val, 4), "0.90 0.90 0.90")
    
    # Vector of expected recruitment over 30 years
    expected_recruits <- round(N * Age_props[1] * (lam^(0:29)))
    
    # spip expects "var" followed by sizes
    cohort_str <- paste("var", paste(expected_recruits, collapse=" "))
    
    # Package inputs
    inputs <- list(
      N_init = N,
      curMort = curMort,
      L = L,
      Age_props = Age_props,
      n_yrs = 30,
      n_ages = 5,
      N_yrs_sample = c(21:30),
      samples_per_year = 25,
      Repro = c(0,0,0,1,1),
      mat_age = 2,
      init_str = init_str,
      surv_str = surv_str,
      cohort_str = cohort_str
    )
    
    # --- 1. fishSim Benchmark ---
    cat("  Running fishSim... ")
    cl <- makeCluster(max_cores)
    registerDoParallel(cl)
    start_time <- Sys.time()
    out <- foreach(i = 1:total_cpu_sims, .packages = c("fishSim")) %dopar% {
      run_fishsim_rep(inputs)
    }
    end_time <- Sys.time()
    stopCluster(cl)
    fs_sims_sec <- total_cpu_sims / as.numeric(difftime(end_time, start_time, units="secs"))
    cat(sprintf("%.2f sims/sec\n", fs_sims_sec))
    
    # --- 2. CKMRpop Benchmark ---
    cat("  Running CKMRpop... ")
    cl <- makeCluster(max_cores)
    registerDoParallel(cl)
    start_time <- Sys.time()
    out <- foreach(i = 1:total_cpu_sims, .packages = c("CKMRpop")) %dopar% {
      run_ckmrpop_rep(inputs)
    }
    end_time <- Sys.time()
    stopCluster(cl)
    ckmr_sims_sec <- total_cpu_sims / as.numeric(difftime(end_time, start_time, units="secs"))
    cat(sprintf("%.2f sims/sec\n", ckmr_sims_sec))
    
    # --- 3. PyTorch GPU Benchmark ---
    cat("  Running PyTorch GPU... ")
    gpu_sims_sec <- py$run_gpu_benchmark_scenario(
      n_sims=1000L, 
      batch_size=1000L, 
      n_init_val=as.integer(N), 
      surv0_val=as.numeric(surv0_val)
    )
    cat(sprintf("%.2f sims/sec\n\n", gpu_sims_sec))
    
    # MODIFIED: Append wide format row to results_df once per scenario
    results_df <- rbind(results_df, data.frame(
      N = N,
      lambda = lam,
      `fishsim sims/sec` = fs_sims_sec,
      `CKMRpop sims/sec` = ckmr_sims_sec,
      `Python/GPU sims/sec` = gpu_sims_sec,
      check.names = FALSE
    ))
  }
}

cat("==========================================\n")
cat("FINAL RESULTS\n")
cat("==========================================\n")
print(results_df)

print(xtable::xtable(results_df,digits=2),include.rownames=FALSE)