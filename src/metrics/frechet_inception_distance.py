from pytorch_fid.fid_score import calculate_fid_given_paths

BATCH_SIZE = 50
NUM_WORKERS = 8

def calculate_fretchet(images_real, images_fake, device='cpu', dims=2048, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    return calculate_fid_given_paths((images_real, images_fake), batch_size, device, dims, num_workers)