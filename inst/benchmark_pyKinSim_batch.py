import torch
import time

def calculate_worst_case_max_pop(n_init_max, surv0_max, age_props, n_yrs):
    """
    Deterministically projects the worst-case cumulative population size using upper bounds
    to safely establish M (max_pop) for PyTorch tensor allocation. 
    It tracks EVERY individual ever born since indices are not recycled.
    """
    N = torch.zeros(5, dtype=torch.float32)
    N[:] = n_init_max * age_props.cpu() 
    
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

def calculate_max_batch_size(max_pop, total_sampled, device):
    """
    Calculates the maximum safe PyTorch batch size (B) based on VRAM limits.
    """
    if device.type == "cpu":
        return 100 
        
    total_vram = torch.cuda.get_device_properties(device).total_memory
    reserved_vram = 1.5 * (1024**3) 
    usable_vram = max(0, total_vram - reserved_vram)
    
    # Persistent State Tensors: alive(1)+ages(1)+sexes(1)+mother_id(2)+father_id(2)+mtdna(1) = 8B
    state_bytes = 8 * max_pop
    
    # Peak Simulation Transient Tensors
    sim_transient_bytes = 8 * max_pop
    
    # Peak Kinship Transient Tensors 
    kinship_transient_bytes = 5 * (total_sampled ** 2) * 1 
    
    total_bytes_per_universe = state_bytes + sim_transient_bytes + kinship_transient_bytes
    
    if total_bytes_per_universe == 0:
        return 100
        
    optimal_B = int(usable_vram / total_bytes_per_universe)
    return optimal_B

def run_gpu_benchmark(device_name="cuda", batch_sizes=None):
    if not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU, but benchmarking requires a GPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
        print(f"Using Device: {torch.cuda.get_device_name(device)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB\n")

    # Demographic parameters
    n_yrs = 30
    n_ages = 5
    
    # ABC Prior Bounds
    n_init_min = 300
    n_init_max = 300
    surv0_min = 0.649   #0.516  
    surv0_max = 0.651   #0.808  
    
    age_props = torch.tensor([0.435, 0.283, 0.141, 0.071, 0.070], device=device)
    mtdna_props = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device) # 4 Haplotypes
    
    samples_per_yr = 25
    sample_yrs = 10 # Years 21-30
    total_sampled = samples_per_yr * sample_yrs
    
    max_pop = calculate_worst_case_max_pop(n_init_max, surv0_max, age_props, n_yrs)
    print(f"Calculated worst-case max_pop (M): {max_pop}")

    # Set default batch sizes if none are provided
    if batch_sizes is None:
        batch_sizes = [10, 50, 100, 250, 500]
    
    print(f"Using user-defined batch sizes (B): {batch_sizes}\n")

    print(f"{'Batch Size':<11} | {'Sims/Sec':<10} | {'Time (s)':<9} | {'VRAM (MB)':<10} | {'POP_M':<7} | {'POP_P':<7} | {'2nd_ShareMT':<11} | {'2nd_DiffMT':<11}")
    print("-" * 105)

    for B in batch_sizes:
        try:
            if device.type == "cuda":
                try:
                    if hasattr(torch.cuda, "reset_peak_memory_stats"):
                        torch.cuda.reset_peak_memory_stats()
                    elif hasattr(torch.cuda, "reset_max_memory_allocated"):
                        torch.cuda.reset_max_memory_allocated()
                except Exception:
                    pass 
                torch.cuda.empty_cache()

            start_time = time.time()

            # ---------------------------------------------------------
            # 1. INITIALIZE MASSIVE TENSORS WITH PRIORS
            # ---------------------------------------------------------
            batch_n_init = torch.randint(n_init_min, n_init_max + 1, (B,), device=device)
            batch_surv0 = (surv0_max - surv0_min) * torch.rand((B,), device=device) + surv0_min
            
            surv_probs = torch.tensor([0.0, 0.90, 0.90, 0.90, 0.0], device=device).unsqueeze(0).repeat(B, 1)
            surv_probs[:, 0] = batch_surv0

            seq = torch.arange(max_pop, device=device).unsqueeze(0).expand(B, -1)
            alive = seq < batch_n_init.unsqueeze(1)
            
            ages = torch.zeros((B, max_pop), dtype=torch.int8, device=device)
            founder_ages = torch.multinomial(age_props, B * max_pop, replacement=True).view(B, max_pop)
            ages = torch.where(alive, founder_ages.to(torch.int8), ages)

            sexes = torch.zeros((B, max_pop), dtype=torch.bool, device=device)
            random_sex = torch.rand((B, max_pop), device=device) > 0.5
            sexes = torch.where(alive, random_sex, sexes)

            mtdna = torch.zeros((B, max_pop), dtype=torch.int8, device=device)
            founder_mtdna = torch.multinomial(mtdna_props, B * max_pop, replacement=True).view(B, max_pop)
            mtdna = torch.where(alive, founder_mtdna.to(torch.int8), mtdna)

            mother_id = torch.full((B, max_pop), -1, dtype=torch.int16, device=device)
            father_id = torch.full((B, max_pop), -1, dtype=torch.int16, device=device)
            
            next_idx = batch_n_init.clone().to(torch.long)

            samples = torch.full((B, total_sampled), -1, dtype=torch.long, device=device)
            capture_yrs = torch.full((B, total_sampled), -1, dtype=torch.int16, device=device)
            s_col = 0

            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_pop)

            # ---------------------------------------------------------
            # 2. VECTORIZED FORWARD SIMULATION
            # ---------------------------------------------------------
            for yr in range(n_yrs):
                # --- A. Survival ---
                rand_surv = torch.rand((B, max_pop), device=device)
                safe_ages = torch.clamp(ages, 0, n_ages - 1).long()
                
                thresholds = surv_probs[batch_idx, safe_ages]
                
                died = alive & (rand_surv > thresholds)
                alive[died] = False
                
                # --- B. Aging ---
                ages[alive] += 1
                max_age_mask = alive & (ages >= n_ages)
                alive[max_age_mask] = False

                # --- C. Sampling (Years 21-30) ---
                if yr >= (n_yrs - sample_yrs):
                    rand_w = torch.rand((B, max_pop), device=device)
                    rand_w[~alive] = -1.0
                    
                    _, sampled_idx = torch.topk(rand_w, samples_per_yr, dim=1)
                    
                    valid_samp_mask = torch.gather(alive, 1, sampled_idx)
                    samples[:, s_col:s_col+samples_per_yr] = torch.where(valid_samp_mask, sampled_idx, -1)
                    
                    capture_yrs[:, s_col:s_col+samples_per_yr] = torch.where(valid_samp_mask, yr, -1)
                    s_col += samples_per_yr

                # --- D. Reproduction ---
                mat_f = alive & sexes & (ages >= 3)
                mat_m = alive & ~sexes & (ages >= 3)
                
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
                    sexes[b_idx_safe, i_idx_safe] = torch.rand(len(b_idx_safe), device=device) > 0.5
                    
                    moms_for_newborns = chosen_moms[valid_newborns][valid_insert]
                    dads_for_newborns = chosen_dads[valid_newborns][valid_insert]
                    
                    mother_id[b_idx_safe, i_idx_safe] = moms_for_newborns.to(torch.int16)
                    father_id[b_idx_safe, i_idx_safe] = dads_for_newborns.to(torch.int16)
                    
                    mtdna[b_idx_safe, i_idx_safe] = mtdna[b_idx_safe, moms_for_newborns.long()]
                    
                    next_idx += num_newborns

            if device.type == "cuda":
                torch.cuda.synchronize()
            
            max_idx_generated = next_idx.max().item()
            if max_idx_generated > max_pop:
                print("\n" + "!" * 80)
                print(f"FATAL ERROR: Buffer safety limit breached at Batch Size {B}.")
                print(f"Generated {max_idx_generated} individuals, but max_pop is {max_pop}.")
                break
                

            # ---------------------------------------------------------
            # 3. KINSHIP EVALUATION (Memory Optimized & mtDNA observable)
            # ---------------------------------------------------------
            dummy_idx = max_pop
            
            samples_safe = samples.clone().long()
            samples_safe[samples_safe == -1] = 0
            
            M = torch.gather(mother_id, 1, samples_safe)
            M[samples == -1] = -1
            
            F = torch.gather(father_id, 1, samples_safe)
            F[samples == -1] = -1
            
            mtdna_samp = torch.gather(mtdna, 1, samples_safe)
            
            M_safe = M.clone().long()
            M_safe[M_safe == -1] = 0 
            
            F_safe = F.clone().long()
            F_safe[F_safe == -1] = 0
            
            MM = torch.gather(mother_id, 1, M_safe)
            MM[M == -1] = -1
            
            FM = torch.gather(father_id, 1, M_safe)
            FM[M == -1] = -1
            
            MF = torch.gather(mother_id, 1, F_safe)
            MF[F == -1] = -1
            
            FF = torch.gather(father_id, 1, F_safe)
            FF[F == -1] = -1

            I_2d = samples.to(torch.int16)
            I_2d[I_2d == -1] = dummy_idx
            
            M_2d = M.to(torch.int16)
            M_2d[M_2d == -1] = dummy_idx
            F_2d = F.to(torch.int16)
            F_2d[F_2d == -1] = dummy_idx
            
            MM_2d = MM.to(torch.int16)
            MM_2d[MM_2d == -1] = dummy_idx
            FM_2d = FM.to(torch.int16)
            FM_2d[FM_2d == -1] = dummy_idx
            MF_2d = MF.to(torch.int16)
            MF_2d[MF_2d == -1] = dummy_idx
            FF_2d = FF.to(torch.int16)
            FF_2d[FF_2d == -1] = dummy_idx

            S_sampled = total_sampled
            triu_mask = torch.triu(torch.ones(S_sampled, S_sampled, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0)
            
            valid_pairs = (I_2d.unsqueeze(2) != dummy_idx) & (I_2d.unsqueeze(1) != dummy_idx) & triu_mask
            
            share_mtdna = (mtdna_samp.unsqueeze(2) == mtdna_samp.unsqueeze(1))

            # --- POPs (Maternal vs Paternal) ---
            pop_mat_mask = ((I_2d.unsqueeze(2) == M_2d.unsqueeze(1)) | (I_2d.unsqueeze(1) == M_2d.unsqueeze(2))) & valid_pairs
            pop_pat_mask = ((I_2d.unsqueeze(2) == F_2d.unsqueeze(1)) | (I_2d.unsqueeze(1) == F_2d.unsqueeze(2))) & valid_pairs
            
            total_pop_mat = pop_mat_mask.sum(dim=(1, 2))
            total_pop_pat = pop_pat_mask.sum(dim=(1, 2))
            del pop_mat_mask, pop_pat_mask
            
            # --- Second Degree Relatives (HSP + FTP + GG) ---
            share_M = (M_2d.unsqueeze(2) == M_2d.unsqueeze(1)) & (M_2d.unsqueeze(2) != dummy_idx)
            share_F = (F_2d.unsqueeze(2) == F_2d.unsqueeze(1)) & (F_2d.unsqueeze(2) != dummy_idx)
            hsp_mask = ((share_M & ~share_F) | (~share_M & share_F))
            
            ftp_mask = (M_2d.unsqueeze(2) == MM_2d.unsqueeze(1)) & (F_2d.unsqueeze(2) == FM_2d.unsqueeze(1)) & (M_2d.unsqueeze(2) != dummy_idx) & (I_2d.unsqueeze(2) != M_2d.unsqueeze(1))
            ftp_mask |= (M_2d.unsqueeze(2) == MF_2d.unsqueeze(1)) & (F_2d.unsqueeze(2) == FF_2d.unsqueeze(1)) & (M_2d.unsqueeze(2) != dummy_idx) & (I_2d.unsqueeze(2) != F_2d.unsqueeze(1))
            ftp_mask |= (M_2d.unsqueeze(1) == MM_2d.unsqueeze(2)) & (F_2d.unsqueeze(1) == FM_2d.unsqueeze(2)) & (M_2d.unsqueeze(1) != dummy_idx) & (I_2d.unsqueeze(1) != M_2d.unsqueeze(2))
            ftp_mask |= (M_2d.unsqueeze(1) == MF_2d.unsqueeze(2)) & (F_2d.unsqueeze(1) == FF_2d.unsqueeze(2)) & (M_2d.unsqueeze(1) != dummy_idx) & (I_2d.unsqueeze(1) != F_2d.unsqueeze(2))
            
            gg_mask = (I_2d.unsqueeze(2) == MM_2d.unsqueeze(1)) | (I_2d.unsqueeze(2) == MF_2d.unsqueeze(1)) | (I_2d.unsqueeze(2) == FM_2d.unsqueeze(1)) | (I_2d.unsqueeze(2) == FF_2d.unsqueeze(1))
            gg_mask |= (I_2d.unsqueeze(1) == MM_2d.unsqueeze(2)) | (I_2d.unsqueeze(1) == MF_2d.unsqueeze(2)) | (I_2d.unsqueeze(1) == FM_2d.unsqueeze(2)) | (I_2d.unsqueeze(1) == FF_2d.unsqueeze(2))
            
            second_degree_mask = (hsp_mask | ftp_mask | gg_mask) & valid_pairs
            
            # --- Split by Observable mtDNA sharing ---
            sd_share_mt_mask = second_degree_mask & share_mtdna
            sd_diff_mt_mask = second_degree_mask & ~share_mtdna
            
            total_sd_share = sd_share_mt_mask.sum(dim=(1, 2))
            total_sd_diff = sd_diff_mt_mask.sum(dim=(1, 2))

            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            elapsed = end_time - start_time
            sims_per_sec = B / elapsed
            
            if device.type == "cuda":
                try:
                    if hasattr(torch.cuda, "max_memory_allocated"):
                        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    else:
                        vram_mb = 0.0
                except Exception:
                    vram_mb = 0.0
            else:
                vram_mb = 0.0
            
            mean_pop_m = total_pop_mat.float().mean().item()
            mean_pop_p = total_pop_pat.float().mean().item()
            mean_sd_share = total_sd_share.float().mean().item()
            mean_sd_diff = total_sd_diff.float().mean().item()
            
            print(f"{B:<11} | {sims_per_sec:<10.0f} | {elapsed:<9.3f} | {vram_mb:<10.1f} | {mean_pop_m:<7.2f} | {mean_pop_p:<7.2f} | {mean_sd_share:<11.2f} | {mean_sd_diff:<11.2f}")

        except Exception as e:
            if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                print("-" * 80)
                print(f"CUDA OUT OF MEMORY (or Async Error) at Batch Size: {B}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            else:
                raise e

if __name__ == "__main__":
    # Define your custom batch sizes here
    run_gpu_benchmark(batch_sizes=[1000, 2000, 4000, 6000, 8000, 10000])
