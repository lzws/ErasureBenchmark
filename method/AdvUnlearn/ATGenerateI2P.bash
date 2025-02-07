   export MODEL_NAME='SD-v1-4'
   export ERASE_CONCEPT='nudity'
   export PROMPTS_PATH='/home/users/diffusion/project/EraseConceptBenchmark/dataset/i2p_4703.csv'
   export save_path='/home/users/diffusion/project/EraseConceptBenchmark/image_result/nsfw/nudity/AdvUnlearn/i2p'
   export device='cuda:0'
   export ddim_steps=40

   nohup python ATPipeline.py \
   --model_name $MODEL_NAME \
   --erase_concept $ERASE_CONCEPT \
   --prompts_path $PROMPTS_PATH \
   --save_path $save_path \
   --device $device \
   --ddim_steps $ddim_steps \
   > ATNudityI2p.log 2>&1 &