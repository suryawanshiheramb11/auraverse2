def load_data(data_path):
    """
    Scans for subsets:
    - YouTube-real (Real)
    - Celeb-real (Real)
    - Celeb-synthesis (Fake)
    """
    real_files = []
    fake_files = []
    
    # Define known folders
    real_folders = ["YouTube-real", "Celeb-real"]
    fake_folders = ["Celeb-synthesis"]
    
    # Scan Real
    for folder in real_folders:
        path = os.path.join(data_path, folder)
        if os.path.exists(path):
            found = glob.glob(os.path.join(path, "*.mp4")) + glob.glob(os.path.join(path, "*.jpg"))
            real_files.extend(found)
            logger.info(f"Found {len(found)} Real samples in {folder}")
            
    # Scan Fake
    for folder in fake_folders:
        path = os.path.join(data_path, folder)
        if os.path.exists(path):
            found = glob.glob(os.path.join(path, "*.mp4")) + glob.glob(os.path.join(path, "*.jpg"))
            fake_files.extend(found)
            logger.info(f"Found {len(found)} Fake samples in {folder}")

    # Fallback: if structure is different look for 'real'/'fake' or root files
    if not real_files and not fake_files:
         # Try generic recursion? Or user might have loose files. 
         # Based on ls output, we saw 'YouTube-real'.
         pass

    list_ids = real_files + fake_files
    labels = {}
    
    for f in real_files:
        labels[f] = 0
    for f in fake_files:
        labels[f] = 1
        
    logger.info(f"Total: {len(real_files)} Real, {len(fake_files)} Fake.")
    random.shuffle(list_ids)
    return list_ids, labels
