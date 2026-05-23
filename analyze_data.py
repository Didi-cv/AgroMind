import csv
from collections import Counter

csv_path = "data/processed/PlantVillageVQA.csv"

question_types = Counter()
splits = Counter()

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        question_types[row['question_type']] += 1
        splits[row['split']] += 1

print("=" * 40)
print("📊 PlantVillageVQA 数据集分析报告")
print("=" * 40)
print(f"\n总问答对数: {sum(question_types.values()):,}")

print(f"\n📂 数据集划分:")
for split, count in splits.items():
    print(f"  {split}: {count:,} 条")

print(f"\n📝 问题类型分布 (共{len(question_types)}类):")
for qtype, count in question_types.most_common():
    pct = count / sum(question_types.values()) * 100
    print(f"  {qtype}: {count:,} ({pct:.1f}%)")

# 找出包含有效答案的样本
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    diagnosis_samples = [row for row in reader 
                         if row['question_type'] == 'Specific Disease Identification' 
                         and row['split'] == 'train'
                         and 'healthy' not in row['answer'].lower()][:3]

print(f"\n🔬 诊断类问答示例 (前3条):")
for i, sample in enumerate(diagnosis_samples, 1):
    print(f"  [{i}] 图片: {sample['image_id']}")
    print(f"      问题: {sample['question']}")
    print(f"      答案: {sample['answer'][:100]}...")
    print()
