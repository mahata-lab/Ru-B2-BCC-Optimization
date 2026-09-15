import csv

# Define the Chemical Space
refractory_elements = ['Nb', 'Mo', 'Ta', 'W', 'V']
b2_formers = ['Hf', 'Ti', 'Zr', 'Al'] # Primary B2 partners
ru_substitutes = ['Al', 'Cr', 'Cu', 'Si'] # The "Penalty" elements (Reviewer request)

# Open CSV for writing
with open('dft_run_list.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Header: ID, Group, Formula (Approx), Structure, Supercell, Description
    writer.writerow(['Run_ID', 'Group', 'Composition_Formula', 'Structure', 'Atoms', 'Description'])
    
    run_id = 1

    # --- SET 1: BASELINES (Pure Elements) ---
    # Need these for Chemical Potential (Convex Hull)
    all_elements = ['Ru'] + refractory_elements + b2_formers + ru_substitutes
    for el in set(all_elements):
        writer.writerow([f"BASE_{run_id:03}", "Baseline", f"{el}1", "BCC/HCP_GroundState", "1 or 2", f"Reference energy for pure {el}"])
        run_id += 1

    # --- SET 2: BINARY B2s (The Corners) ---
    # Simple 2-atom cells
    for partner in b2_formers:
        writer.writerow([f"BIN_{run_id:03}", "Binary_B2", f"Ru1{partner}1", "B2_Primitive", "2", f"Pure Binary Ru-{partner}"])
        run_id += 1

    # --- SET 3: THE "KUBE GRID" (Matrix Solubility) ---
    # Logic: How much Matrix (Nb, Mo, Ta) dissolves into the B2?
    # Method: 2x2x2 Supercell (16 atoms). 8 Ru atoms fixed. 
    # The other 8 sites are (7 Partner + 1 Matrix) or (6 Partner + 2 Matrix)
    # Represents 12.5% and 25% solubility.
    
    for partner in ['Hf', 'Ti', 'Al']: # The main B2 systems
        for matrix_el in ['Nb', 'Mo', 'Ta', 'V']:
            # 12.5% Solubility (1 atom substitution)
            formula = f"Ru8{partner}7{matrix_el}1"
            writer.writerow([f"KUBE_{run_id:03}", "Solubility_Study", formula, "B2_SQS_16", "16", f"Ru-{partner} B2 with 12.5% {matrix_el} solubility"])
            run_id += 1
            
            # 25% Solubility (2 atom substitution)
            formula = f"Ru8{partner}6{matrix_el}2"
            writer.writerow([f"KUBE_{run_id:03}", "Solubility_Study", formula, "B2_SQS_16", "16", f"Ru-{partner} B2 with 25% {matrix_el} solubility"])
            run_id += 1

    # --- SET 4: THE PENALTY STUDY (Reviewer Demand) ---
    # Logic: What happens if we replace expensive Ru with Al, Cr, Cu?
    # Method: 2x2x2 Supercell. 8 Partner atoms fixed (e.g., Hf).
    # The 8 Ru sites are (7 Ru + 1 Dopant) or (6 Ru + 2 Dopant).
    
    for partner in ['Hf', 'Ti']: # Focus on the most stable B2s for this
        for dopant in ru_substitutes:
            # 12.5% Substitution (Penalty Check)
            formula = f"Ru7{dopant}1{partner}8"
            writer.writerow([f"PEN_{run_id:03}", "Ru_Substitution", formula, "B2_SQS_16", "16", f"Replacing 12.5% Ru with {dopant} in Ru-{partner}"])
            run_id += 1
            
            # 25% Substitution (Severe Penalty Check)
            formula = f"Ru6{dopant}2{partner}8"
            writer.writerow([f"PEN_{run_id:03}", "Ru_Substitution", formula, "B2_SQS_16", "16", f"Replacing 25% Ru with {dopant} in Ru-{partner}"])
            run_id += 1

    # --- SET 5: THE MATRIX (Disordered Reference) ---
    # Logic: We need the energy of the Disordered Matrix to calculate Solvus.
    # Method: 16-atom SQS of pure matrix combinations.
    
    matrix_pairs = [('Nb','Mo'), ('Nb','Ta'), ('Mo','V'), ('Nb','Ti')]
    for (m1, m2) in matrix_pairs:
        # 50/50 Matrix
        formula = f"{m1}8{m2}8"
        writer.writerow([f"MAT_{run_id:03}", "Disordered_Matrix", formula, "BCC_SQS_16", "16", f"Disordered {m1}-{m2} Matrix Reference"])
        run_id += 1
        
        # 75/25 Matrix
        formula = f"{m1}12{m2}4"
        writer.writerow([f"MAT_{run_id:03}", "Disordered_Matrix", formula, "BCC_SQS_16", "16", f"Disordered {m1}-rich Matrix Reference"])
        run_id += 1

    print(f"Successfully generated CSV with {run_id-1} unique DFT runs.")
