# Diagnostic Script: Comparing fishSim (R) vs PyTorch (GPU) Simulations

library(fishSim)
library(spatstat)
library(reticulate)

st_time = Sys.time()

# ------------------------------------------------------------------------------
# EMBEDDED PYTHON GPU SIMULATOR (FIXED PARAMETERS FOR CONSTANT ABUNDANCE)
# ------------------------------------------------------------------------------
cat("Initializing Python GPU environment...\n")
py_code <- "
import torch
import time
import pandas as pd
import sys

def calculate_worst_case_max_pop(n_init_max, surv0_max, n_yrs):
    L = torch.zeros((5, 5), dtype=torch.float64, device='cpu')
    L[0, 2] = 0.9
    L[0, 3] = 0.9
    L[1, 0] = surv0_max
    L[2, 1] = 0.9
    L[3, 2] = 0.9
    L[4, 3] = 0.9
    
    vals, vecs = torch.linalg.eig(L)
    max_idx = vals.real.argmax()
    age_props = vecs[:, max_idx].real
    age_props = (age_props / age_props.sum()).float()

    N = torch.zeros(5, dtype=torch.float32)
    N[:] = n_init_max * age_props 
    
    total_individuals_ever = n_init_max
    S = [surv0_max, 0.90, 0.90, 0.90, 0.0]
    
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
        
    return int(total_individuals_ever * 3.0)

def generate_abc_reference_table(n_sims=1000, batch_size=500, device_name='cuda'):
    if not torch.cuda.is_available():
        print('CUDA not available. Falling back to CPU, but benchmarking requires a GPU.', flush=True)
        device = torch.device('cpu')
    else:
        device = torch.device(device_name)
        print(f'Using Device: {torch.cuda.get_device_name(device)}', flush=True)
        print(f'Generating {n_sims} PyTorch simulations in batches of {batch_size}...', flush=True)

    n_yrs = 20
    n_ages = 5
    
    # FIXED PARAMETERS FOR CONSTANT EXPECTED ABUNDANCE
    n_init_fixed = 200
    surv0_fixed = 0.65  
    
    mtdna_props = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
    
    samples_per_yr = 25
    sample_yrs = 10
    total_sampled = samples_per_yr * sample_yrs
    
    max_pop = calculate_worst_case_max_pop(n_init_fixed, surv0_fixed, n_yrs)

    sims_completed = 0
    start_time_global = time.time()
    
    all_batches = [] 
    
    while sims_completed < n_sims:
        B = min(batch_size, n_sims - sims_completed)
        torch.cuda.empty_cache()

        # FIXED: N_init = 200 for all simulations in the batch
        batch_n_init = torch.full((B,), n_init_fixed, dtype=torch.long, device=device)

        # FIXED: surv0 = 0.65 for all simulations in the batch
        batch_surv0 = torch.full((B,), surv0_fixed, dtype=torch.float32, device=device)
        
        # CHANGED: Aligned survival matrix so Age 1 gets 0.65, Ages 2-4 get 0.90, Age 5 gets 0.0
        surv_probs = torch.tensor([0.0, 0.0, 0.90, 0.90, 0.90, 0.0], device=device).unsqueeze(0).repeat(B, 1)
        surv_probs[:, 1] = batch_surv0

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
        # CHANGED: Founders are now strictly ages 1, 2, 3, 4, 5
        founder_ages = torch.multinomial(age_props_batch, max_pop, replacement=True) + 1
        ages = torch.where(alive, founder_ages.to(torch.int8), ages)

        birth_years = torch.full((B, max_pop), -1, dtype=torch.int8, device=device)
        # CHANGED: Matched fishSim birth year initialization logic
        birth_years = torch.where(alive, (1 - ages).to(torch.int8), birth_years)

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
        N_breed_sum = torch.zeros(B, device=device)

        for yr in range(n_yrs):
            yr_sim = yr + 1
            
            rand_surv = torch.rand((B, max_pop), device=device)
            # CHANGED: Max age clamp is now 5 to match the 6-element survival array
            safe_ages = torch.clamp(ages, 0, 5).long()
            thresholds = surv_probs[batch_idx, safe_ages]
            
            died_this_yr = alive & (rand_surv > thresholds)
            death_years[died_this_yr] = yr_sim
            alive[died_this_yr] = False
            
            # CHANGED: Kill fish explicitly at age 5
            max_age_mask = alive & (ages >= 5)
            death_years[max_age_mask] = yr_sim
            alive[max_age_mask] = False

            # CHANGED: Maturity is now ages 3 and 4
            mat_f = alive & sexes & ((ages == 3) | (ages == 4))
            mat_m = alive & ~sexes & ((ages == 3) | (ages == 4))
            N_breed_sum += (mat_f | mat_m).sum(dim=1).float()

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
                # CHANGED: 1-indexed birth years to match fishSim exactly
                birth_years[b_idx_safe, i_idx_safe] = yr_sim
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
        failed_sims = torch.zeros(B, dtype=torch.bool, device=device)
        
        for sy in range(11, 21):
            valid_dead = (death_years == sy) & (idx_grid < next_idx.unsqueeze(1))
            valid_count = valid_dead.sum(dim=1)
            failed_sims |= (valid_count <= 5)
            
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

        if device.type == 'cuda':
            torch.cuda.synchronize()

        mean_N_breed = N_breed_sum / n_yrs
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
            # CHANGED: 1-based bounds for o_by
            valid_match = valid_p & (p_samp_yr >= 11) & (o_by >= 1) & (o_by <= n_yrs)
            
            b_match = B_idx[valid_match]
            p_sy_match = (p_samp_yr[valid_match] - 11).long()
            o_by_match = o_by[valid_match].long() - 1  # -1 map to matrix column index
            
            flat_idx = b_match * (sample_yrs * n_yrs) + p_sy_match * n_yrs + o_by_match
            po_matrix_flat.scatter_add_(0, flat_idx.long(), torch.ones_like(flat_idx, dtype=torch.int32))

        accumulate_po(m_ids)
        accumulate_po(f_ids)
        
        samp_sy_flat = torch.repeat_interleave(torch.arange(11, 21, dtype=torch.int8, device=device), samples_per_yr)
        by_i = o_by.unsqueeze(2).expand(B, total_sampled, total_sampled)
        by_j = o_by.unsqueeze(1).expand(B, total_sampled, total_sampled)
        sy_i = samp_sy_flat.unsqueeze(0).unsqueeze(2).expand(B, total_sampled, total_sampled)
        valid_i = valid_samp.unsqueeze(2).expand(B, total_sampled, total_sampled)
        valid_j = valid_samp.unsqueeze(1).expand(B, total_sampled, total_sampled)
        
        # CHANGED: 1-based bounds for by_j
        comp_mask = valid_i & valid_j & (by_j > by_i + 2) & (by_j < sy_i) & (by_j >= 1) & (by_j <= n_yrs)
        
        comp_matrix_flat = torch.zeros(B * sample_yrs * n_yrs, dtype=torch.int32, device=device)
        b_idx_comp = B_idx.unsqueeze(2).expand(B, total_sampled, total_sampled)[comp_mask]
        sy_match_comp = (sy_i[comp_mask] - 11).long()
        by_match_comp = by_j[comp_mask].long() - 1  # -1 map to matrix column index
        
        flat_idx_comp = b_idx_comp * (sample_yrs * n_yrs) + sy_match_comp * n_yrs + by_match_comp
        comp_matrix_flat.scatter_add_(0, flat_idx_comp.long(), torch.ones_like(flat_idx_comp, dtype=torch.int32))
        
        po_matrix_2d = po_matrix_flat.view(B, sample_yrs, n_yrs).float()
        comp_matrix_2d = comp_matrix_flat.view(B, sample_yrs, n_yrs).float()

        # Extract diagnostic summaries (Total Matches, Total Comps, Mean N_breed)
        matches_by = po_matrix_2d.sum(dim=1) 
        comps_by = comp_matrix_2d.sum(dim=1) 
        tot_matches = matches_by.sum(dim=1)
        tot_comps = comps_by.sum(dim=1)

        # ------------------------------------------------------------------
        # ASSEMBLE DIAGNOSTIC DATAFRAME
        # ------------------------------------------------------------------
        mean_N_breed_cpu = mean_N_breed.cpu().numpy()
        tot_matches_cpu = tot_matches.cpu().numpy()
        tot_comps_cpu = tot_comps.cpu().numpy()
        failed_sim_cpu = failed_sims.cpu().numpy().astype(bool)
        
        batch_df = pd.DataFrame({
            'mean_N_breed': mean_N_breed_cpu,
            'tot_matches': tot_matches_cpu,
            'tot_comps': tot_comps_cpu,
            'failed_sim': failed_sim_cpu
        })
        
        all_batches.append(batch_df)
        sims_completed += B

    final_df = pd.concat(all_batches, ignore_index=True)
    return final_df
"
py_run_string(py_code)

# ------------------------------------------------------------------------------
# R fishSim DIAGNOSTIC LOOP 
# ------------------------------------------------------------------------------

set.seed(1111)
n_sim = 1000

mat_age = 2 
N_init = 200
ageMort = 1-c(0.65,0.9,0.9,0.9,0)
L = matrix(0,5,5)
L[1,]=c(0,0,.9,.9,0)
L[2,1]=0.65
L[3,2]=L[4,3]=L[5,4]=0.9

Age_props = Re(eigen(L)$vectors[,1]/sum(eigen(L)$vectors[,1]))
Repro = c(0,0,0,1,1)  

n_ages=5
n_yrs =20
N_yrs_sample = c(11:20)
n_yrs_sample_len = length(N_yrs_sample)
samples_per_year = 25

# DataFrame to hold the results from R fishSim
R_results <- data.frame(
  mean_N_breed = rep(NA, n_sim),
  tot_matches = rep(NA, n_sim),
  tot_comps = rep(NA, n_sim)
)

cat(sprintf("Generating %d R fishSim simulations...\n", n_sim))

for(isim in 1:n_sim){
  if (isim %% 100 == 0) cat(sprintf("R sim %d / %d...\n", isim, n_sim))
  
  flag=0
  N_breed = rep(0,n_yrs)
  archive <- make_archive()
  indiv = makeFounders(pop=N_init,osr=c(0.5,0.5),stocks=1,minAge=1,maxAge=5,survCurv=Age_props)
  
  for(iyr in 1:n_yrs){
    indiv <- mort(indiv = indiv, type = "age", ageMort = c(0,ageMort), year = iyr,maxAge=n_ages)
    archive <- archive_dead(indiv = indiv, archive = archive)
    indiv <- remove_dead(indiv = indiv)
    N_breed[iyr] = sum(indiv$AgeLast %in% c(3,4))
    indiv <- altMate(indiv = indiv, type='ageSex',maleCurve=Repro,femaleCurve=Repro,osr = c(0.5,0.5),batchSize=2, fecundityDist="poisson",year = iyr)
    indiv <- birthdays(indiv = indiv)
  }
  
  R_results$mean_N_breed[isim] <- mean(N_breed)
  
  # sample from archive (dead only)
  for(isampyr in 1:n_yrs_sample_len){
    cur_yr = N_yrs_sample[isampyr]
    Cur_which = which(archive$DeathY==cur_yr)
    if(length(Cur_which)>=samples_per_year){
      archive[sample(Cur_which,samples_per_year),"SampY"]=N_yrs_sample[isampyr]
    }
    else flag=1
  }
  
  if(flag==0){
    indiv=rbind(archive,indiv)
    POPs <- quickin(indiv,max_gen=2)[["POP"]]
    
    n_comp_PO_sidibj=n_match_PO_sidibj=array(0,dim=c(2,n_yrs,n_yrs))  
    
    if (!is.null(POPs) && nrow(POPs) > 0) {
      for(i in 1:nrow(POPs)){
        indiv1 = indiv[which(indiv[,1]==POPs[i,1]),]
        indiv2 = indiv[which(indiv[,1]==POPs[i,2]),]
        
        FIRST = (indiv1$BirthY < indiv2$BirthY)
        if(FIRST){
          p_by = indiv1$BirthY
          p_sex = (indiv1$Sex=="M")+1  
          p_sy = indiv1$SampY
          o_by = indiv2$BirthY  
        } else {
          p_by = indiv2$BirthY
          p_sex = (indiv2$Sex=="M")+1  
          p_sy = indiv2$SampY
          o_by = indiv1$BirthY  
        }
        n_match_PO_sidibj[p_sex,p_sy,o_by]=n_match_PO_sidibj[p_sex,p_sy,o_by]+1 
      }
    }
    
    Sampled = indiv[is.na(indiv$SampY)==FALSE,]
    n_indiv = nrow(Sampled)
    if (n_indiv > 1) {
      for(i1 in 1:(n_indiv-1)){
        for(i2 in (i1+1):n_indiv){
          born_diff = Sampled$BirthY[i1] - Sampled$BirthY[i2]
          if(born_diff<0){ 
            p_by = Sampled$BirthY[i1]
            p_sy = Sampled$SampY[i1]
            p_sex = (Sampled$Sex[i1]=="M")+1  
            o_by = Sampled$BirthY[i2]   
          } else if(born_diff>0) { 
            p_by = Sampled$BirthY[i2]
            p_sy = Sampled$SampY[i2]
            p_sex = (Sampled$Sex[i2]=="M")+1  
            o_by = Sampled$BirthY[i1]   
          }
          if(born_diff!=0 && o_by>(p_by+mat_age) && o_by<p_sy){
            n_comp_PO_sidibj[p_sex,p_sy,o_by]=n_comp_PO_sidibj[p_sex,p_sy,o_by]+1 
          }
        }
      }
    }
    
    n_comp_PO_dibj = apply(n_comp_PO_sidibj,c(2,3),'sum')
    n_match_PO_dibj = apply(n_match_PO_sidibj,c(2,3),'sum')
    
    # Store totals for years 11:20 to match python bounds exactly
    R_results$tot_matches[isim] <- sum(n_match_PO_dibj[11:20,])
    R_results$tot_comps[isim] <- sum(n_comp_PO_dibj[11:20,])
  } else {
    R_results$tot_matches[isim] <- NA
    R_results$tot_comps[isim] <- NA
  }
}

# ------------------------------------------------------------------------------
# RUN PYTHON GPU SIMULATOR 
# ------------------------------------------------------------------------------
Py_results <- py$generate_abc_reference_table(n_sims=1000L, batch_size=500L)

# ------------------------------------------------------------------------------
# SYSTEMATIC COMPARISON
# ------------------------------------------------------------------------------
cat("\n\n=======================================================\n")
cat("SIMULATION DIAGNOSTIC COMPARISON (R vs Python GPU)\n")
cat("=======================================================\n\n")

cat("--- MEAN BREEDER ABUNDANCE (N_breed_mean) ---\n")
cat(sprintf("R fishSim : Mean = %.2f, SD = %.2f\n", 
            mean(R_results$mean_N_breed, na.rm=TRUE), 
            sd(R_results$mean_N_breed, na.rm=TRUE)))
cat(sprintf("PyTorch   : Mean = %.2f, SD = %.2f\n\n", 
            mean(Py_results$mean_N_breed, na.rm=TRUE), 
            sd(Py_results$mean_N_breed, na.rm=TRUE)))

cat("--- TOTAL KIN PAIR MATCHES (tot_matches) ---\n")
cat(sprintf("R fishSim : Mean = %.2f, SD = %.2f\n", 
            mean(R_results$tot_matches, na.rm=TRUE), 
            sd(R_results$tot_matches, na.rm=TRUE)))
cat(sprintf("PyTorch   : Mean = %.2f, SD = %.2f\n\n", 
            mean(Py_results$tot_matches, na.rm=TRUE), 
            sd(Py_results$tot_matches, na.rm=TRUE)))

cat("--- TOTAL KIN PAIR COMPARISONS (tot_comps) ---\n")
cat(sprintf("R fishSim : Mean = %.2f, SD = %.2f\n", 
            mean(R_results$tot_comps, na.rm=TRUE), 
            sd(R_results$tot_comps, na.rm=TRUE)))
cat(sprintf("PyTorch   : Mean = %.2f, SD = %.2f\n\n", 
            mean(Py_results$tot_comps, na.rm=TRUE), 
            sd(Py_results$tot_comps, na.rm=TRUE)))

cat(sprintf("Total Execution Time: %.2f mins\n", as.numeric(Sys.time()-st_time, units="mins")))