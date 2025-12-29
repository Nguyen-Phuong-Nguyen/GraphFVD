# Script to extract 50 samples from my_train.jsonl for analysis
import json
import random

# input_file = "/kaggle/working/GraphFVD/dataset/NVD/my_train.jsonl"
# output_file = "/kaggle/working/GraphFVD/dataset/NVD/samples_50.jsonl"
input_file = "./dataset/NVD/my_train.jsonl"
output_file = "./dataset/NVD/samples_50.jsonl"

# Read all data
data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

print(f"Total samples: {len(data)}")

# Random sample 50
random.seed(42)
samples = random.sample(data, min(50, len(data)))

# Count labels
label_0 = sum(1 for s in samples if s.get('target', 0) == 0)
label_1 = sum(1 for s in samples if s.get('target', 0) == 1)
print(f"Sampled: {len(samples)} (label 0: {label_0}, label 1: {label_1})")

# Save samples
with open(output_file, 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps(sample) + '\n')

print(f"Saved to: {output_file}")

# Print summary of first sample
print("\n" + "="*60)
print("First sample structure:")
print("="*60)
sample = samples[0]
print(f"Keys: {list(sample.keys())}")
print(f"idx: {sample.get('idx', 'N/A')}")
print(f"target: {sample.get('target', 'N/A')}")
print(f"num nodes: {len(sample.get('nodes_codes', []))}")
print(f"num edges: {len(sample.get('edges', []))}")

# Print first 3 node codes
print(f"\nFirst 3 node codes:")
for i, code in enumerate(sample.get('nodes_codes', [])[:3]):
    print(f"  [{i}]: {code[:100]}..." if len(code) > 100 else f"  [{i}]: {code}")

# Print first 5 edges
print(f"\nFirst 5 edges:")
for i, edge in enumerate(sample.get('edges', [])[:5]):
    print(f"  {edge}")
