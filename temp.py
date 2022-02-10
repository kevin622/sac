import pickle
save_path = 'checkpoints/buffer_temp'
a = 300

with open(save_path, 'wb') as f:
    pickle.dump(a, f)

with open(save_path, 'rb') as f:
    b = pickle.load(f)
print(b)
