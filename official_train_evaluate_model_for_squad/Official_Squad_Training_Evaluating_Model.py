#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('sh run_squad_evaluate_without_negative.sh')


# Answering for predictions without further training.
# 12/07/2020 07:10:15 - INFO - __main__ -   Loading features from cached file ./cached_dev_checkpoint-32000_384
# 12/07/2020 07:10:24 - INFO - __main__ -   ***** Running evaluation  *****
# 12/07/2020 07:10:24 - INFO - __main__ -     Num examples = 12232
# 12/07/2020 07:10:24 - INFO - __main__ -     Batch size = 8
# Evaluating: 100%|███████████████████████████| 1529/1529 [30:19<00:00,  1.19s/it]
# 12/07/2020 07:40:44 - INFO - __main__ -     Evaluation done in total 1819.866386 secs (0.148779 sec per example)
# [INFO|squad_metrics.py:389] 2020-12-07 07:40:44,054 >> Writing predictions to: ./output_without_negative/predictions_.json
# [INFO|squad_metrics.py:391] 2020-12-07 07:40:44,054 >> Writing nbest to: ./output_without_negative/nbest_predictions_.json
# 12/07/2020 07:41:41 - INFO - __main__ -   Results: {'exact': 39.67826160195401, 'f1': 43.86554553616089, 'total': 11873, 'HasAns_exact': 79.47031039136303, 'HasAns_f1': 87.85688632773923, 'HasAns_total': 5928, 'NoAns_exact': 0.0, 'NoAns_f1': 0.0, 'NoAns_total': 5945, 'best_exact': 50.11370336056599, 'best_exact_thresh': 0.0, 'best_f1': 50.11370336056599, 'best_f1_thresh': 0.0}

# In[ ]:


get_ipython().system('sh run_squad_evaluate_with_negative.sh')


# convert squad examples to features: 100%|█| 11873/11873 [00:35<00:00, 331.56it/s
# add example index and unique id: 100%|█| 11873/11873 [00:00<00:00, 587293.57it/s
# 12/08/2020 01:46:25 - INFO - __main__ -   Saving features into cached file ./cached_dev_checkpoint-30000_384
# 12/08/2020 01:46:40 - INFO - __main__ -   ***** Running evaluation  *****
# 12/08/2020 01:46:40 - INFO - __main__ -     Num examples = 12232
# 12/08/2020 01:46:40 - INFO - __main__ -     Batch size = 8
# Evaluating: 100%|███████████████████████████| 1529/1529 [30:10<00:00,  1.18s/it]
# 12/08/2020 02:16:51 - INFO - __main__ -     Evaluation done in total 1810.737200 secs (0.148033 sec per example)
# [INFO|squad_metrics.py:389] 2020-12-08 02:16:51,681 >> Writing predictions to: ./output_with_negative/predictions_.json
# [INFO|squad_metrics.py:391] 2020-12-08 02:16:51,681 >> Writing nbest to: ./output_with_negative/nbest_predictions_.json
# [INFO|squad_metrics.py:393] 2020-12-08 02:16:51,681 >> Writing null_log_odds to: ./output_with_negative/null_odds_.json
# 12/08/2020 02:17:45 - INFO - __main__ -   Results: {'exact': 71.48151267581909, 'f1': 74.90992816283496, 'total': 11873, 'HasAns_exact': 71.28879892037787, 'HasAns_f1': 78.15546172019893, 'HasAns_total': 5928, 'NoAns_exact': 71.67367535744323, 'NoAns_f1': 71.67367535744323, 'NoAns_total': 5945, 'best_exact': 71.48151267581909, 'best_exact_thresh': 0.0, 'best_f1': 74.90992816283492, 'best_f1_thresh': 0.0}

# # run_squad_evaluate_with_negative.sh

# In[ ]:


python run_squad.py --model_type bert --model_name_or_path ./output_with_negative/checkpoint-30000 --do_eval --do_lower_case --version_2_with_negative --predict_file ./squad_v2/dev-v2.0.json --per_gpu_eval_batch_size 8 --max_seq_length 384 --doc_stride 128 --output_dir ./output_with_negative --threads 4 --no_cuda


# convert squad examples to features: 100%|█| 11873/11873 [00:35<00:00, 331.56it/s add example index and unique id: 100%|█| 11873/11873 [00:00<00:00, 587293.57it/s 12/08/2020 01:46:25 - INFO - main - Saving features into cached file ./cached_dev_checkpoint-30000_384 12/08/2020 01:46:40 - INFO - main - * Running evaluation * 12/08/2020 01:46:40 - INFO - main - Num examples = 12232 12/08/2020 01:46:40 - INFO - main - Batch size = 8 Evaluating: 100%|███████████████████████████| 1529/1529 [30:10<00:00, 1.18s/it] 12/08/2020 02:16:51 - INFO - main - Evaluation done in total 1810.737200 secs (0.148033 sec per example) [INFO|squad_metrics.py:389] 2020-12-08 02:16:51,681 >> Writing predictions to: ./output_with_negative/predictions_.json [INFO|squad_metrics.py:391] 2020-12-08 02:16:51,681 >> Writing nbest to: ./output_with_negative/nbest_predictions_.json [INFO|squad_metrics.py:393] 2020-12-08 02:16:51,681 >> Writing null_log_odds to: ./output_with_negative/null_odds_.json 12/08/2020 02:17:45 - INFO - main - Results: {'exact': 71.48151267581909, 'f1': 74.90992816283496, 'total': 11873, 'HasAns_exact': 71.28879892037787, 'HasAns_f1': 78.15546172019893, 'HasAns_total': 5928, 'NoAns_exact': 71.67367535744323, 'NoAns_f1': 71.67367535744323, 'NoAns_total': 5945, 'best_exact': 71.48151267581909, 'best_exact_thresh': 0.0, 'best_f1': 74.90992816283492, 'best_f1_thresh': 0.0}

# # run_squad_train_evaluate_with_negative.sh

# In[ ]:


python run_squad.py --model_type bert --model_name_or_path ./pretrained_model/bert-base-uncased --do_train --do_eval --do_lower_case --version_2_with_negative --train_file ./squad_v2/train-v2.0.json --predict_file ./squad_v2/dev-v2.0.json --per_gpu_train_batch_size 12 --per_gpu_eval_batch_size 8 --learning_rate 3e-5 --num_train_epochs 3.0 --max_seq_length 384 --doc_stride 128 --output_dir ./output_with_negative --save_steps 5000 --threads 4


# # official train and evaluate model for squad
# We study and use official template model for squad to get a trained QA model, we get familiar with many useful functions of the source code which can be used in other different kinds of applications. 
# 
# ## Packages
# - All packages from run_squad.py
# 
# ## The following scripts are used to train and evaluate the model:
# - 1) Evaluate the model
# 
# python run_squad.py --model_type bert --model_name_or_path ./output_with_negative/checkpoint-30000 --do_eval --do_lower_case --version_2_with_negative --predict_file ./squad_v2/dev-v2.0.json --per_gpu_eval_batch_size 8 --max_seq_length 384 --doc_stride 128 --output_dir ./output_with_negative --threads 4 --no_cuda
# 
# - 2) Train and evaluate the model
# 
# python run_squad.py --model_type bert --model_name_or_path ./pretrained_model/bert-base-uncased --do_train --do_eval --do_lower_case --version_2_with_negative --train_file ./squad_v2/train-v2.0.json --predict_file ./squad_v2/dev-v2.0.json --per_gpu_train_batch_size 12 --per_gpu_eval_batch_size 8 --learning_rate 3e-5 --num_train_epochs 3.0 --max_seq_length 384 --doc_stride 128 --output_dir ./output_with_negative --save_steps 5000 --threads 4
# 
# 
# ## Pretrained model
# You can download the pretrained weights from the [link](https://huggingface.co/bert-large-uncased-whole-word-masking/tree/main)
# You can also download the pretrained weights from the [link](https://huggingface.co/bert-base-uncased)
# 
# ## Special code
# ```python
# # Prepare optimizer and schedule (linear warmup and decay)
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": args.weight_decay,
#         },
#         {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
#     )
# 
#     # Check if saved optimizer or scheduler states exist
#     if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
#         os.path.join(args.model_name_or_path, "scheduler.pt")
#     ):
#         # Load in optimizer and scheduler states
#         optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
#         scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
# 
# # Save model checkpoint
# if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
#     output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
#     # Take care of distributed/parallel training
#     model_to_save = model.module if hasattr(model, "module") else model
#     model_to_save.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
# 
#     torch.save(args, os.path.join(output_dir, "training_args.bin"))
#     logger.info("Saving model checkpoint to %s", output_dir)
# 
#     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
#     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
#     logger.info("Saving optimizer and scheduler states to %s", output_dir)
# ```

# In[ ]:



