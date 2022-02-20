#!/bin/bash

#task2
batch_sz=(1 2 4 8 16 32 64 128 256 442 50 100 150 200 250 300 350 400)
for b in "${batch_sz[@]}"
do
	echo "batch size of $b"
	rm -rf /home/cc/.local/share/deepspeech/checkpoints/
	/usr/local/cuda-10.0/bin/nvprof  -f -o nvvp/nvprof/nvprof_baseline_half_batch$b.nvvp --unified-memory-profiling off --metrics all --profile-from-start off python3 DeepSpeech/DeepSpeech.py --train_files set/clips/half_train.csv --dev_files set/clips/half_dev.csv --train_batch_size=$b --dev_batch_size=$b | tee logs/nvprof/nvprof_baseline_half_batch$b.log
	rm -rf /home/cc/.local/share/deepspeech/checkpoints/
	nsys profile --stats=true -o nvvp/nsys/nsys_baseline_half_batch$b python3 DeepSpeech/DeepSpeech.py --train_files set/clips/half_train.csv --dev_files set/clips/half_dev.csv --train_batch_size=$b --dev_batch_size=$b | tee logs/nsys/nsys_baseline_half_batch$b.log
done

#task3
echo "mixed prec"
rm -rf /home/cc/.local/share/deepspeech/checkpoints/
/usr/local/cuda-10.0/bin/nvprof  -f -o nvvp/nvprof/nvprof_baseline_half_batch256_mixFP.nvvp --unified-memory-profiling off --metrics all --profile-from-start off python3 DeepSpeech/DeepSpeech.py --train_files set/clips/half_train.csv --dev_files set/clips/half_dev.csv --train_batch_size=256 --dev_batch_size=256 --automatic_mixed_precision | tee logs/nvprof/nvprof_baseline_half_batch256_mixFP.log
rm -rf /home/cc/.local/share/deepspeech/checkpoints/
nsys profile --stats=true -o nvvp/nsys/nsys_baseline_half_batch256_mixFP python3 DeepSpeech/DeepSpeech.py --train_files set/clips/half_train.csv --dev_files set/clips/half_dev.csv --train_batch_size=256 --dev_batch_size=256 --automatic_mixed_precision | tee logs/nsys/nsys_baseline_half_batch256_mixFP.log

FPs=(16 32 64)
for fp in "${fps[@]}"
do
	echo "fp $fp"
        rm -rf /home/cc/.local/share/deepspeech/checkpoints/
        /usr/local/cuda-10.0/bin/nvprof  -f -o nvvp/nvprof/nvprof_baseline_half_batch256_FP$fp.nvvp --unified-memory-profiling off --metrics all --profile-from-start off python3 DeepSpeech/DeepSpeech.py --train_files set/clips/half_train.csv --dev_files set/clips/half_dev.csv --train_batch_size=256 --dev_batch_size=256 --FP=$fp | tee logs/nvprof/nvprof_baseline_half_batch256_FP$fp.log
        rm -rf /home/cc/.local/share/deepspeech/checkpoints/
        nsys profile --stats=true -o nvvp/nsys/nsys_baseline_half_batch256_FP$fp python3 DeepSpeech/DeepSpeech.py --train_files set/clips/half_train.csv --dev_files set/clips/half_dev.csv --train_batch_size=256 --dev_batch_size=256 --FP=$fp | tee logs/nsys/nsys_baseline_half_batch256_FP$fp.log
done

#task4
/usr/local/cuda-10.0/bin/nvprof  -f -o nvvp/nvprof/nvprof_infer.nvpp deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio audio/4507-16021-0012.wav | tee logs/nvprof/nvprof_infer.log
nsys profile --stats=true -o nvvp/nsys/nsys_infer deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio audio/4507-16021-0012.wav | tee logs/nsys/nsys_infer.log

#optional+ncu
ncu --target-processes all --section Occupancy --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis -o nvvp/ncu/infer.nvvp deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio audio/4507-16021-0012.wav | tee logs/nvvp/nvvp_infer.log
ncu --target-processes all --section Occupancy --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis -o nvvp/ncu/large_batch.nvvp python3 DeepSpeech/DeepSpeech.py --train_files set/clips/small_train.csv --dev_files set/clips/small_dev.csv --train_batch_size=256 --dev_batch_size=256 | tee logs/ncu/largebatch.log
ncu --target-processes all --section Occupancy --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis -o nvvp/ncu/large_batch.nvvp python3 DeepSpeech/DeepSpeech.py --train_files set/clips/small_train.csv --dev_files set/clips/small_dev.csv --train_batch_size=256 --dev_batch_size=256 --FP=64 | tee logs/ncu/largebatch_FP64.log
