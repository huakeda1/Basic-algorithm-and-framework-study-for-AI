# official train and evaluate model for squad
We study and use official template model for squad to get a trained QA model, we get familiar with many useful functions of the source code which can be used in other different kinds of applications. 

## Packages
- All packages from run_squad.py

## The following scripts are used to train and evaluate the model:
- 1) Evaluate the model

python run_squad.py --model_type bert --model_name_or_path ./output_with_negative/checkpoint-30000 --do_eval --do_lower_case --version_2_with_negative --predict_file ./squad_v2/dev-v2.0.json --per_gpu_eval_batch_size 8 --max_seq_length 384 --doc_stride 128 --output_dir ./output_with_negative --threads 4 --no_cuda

- 2) Train and evaluate the model

python run_squad.py --model_type bert --model_name_or_path ./pretrained_model/bert-base-uncased --do_train --do_eval --do_lower_case --version_2_with_negative --train_file ./squad_v2/train-v2.0.json --predict_file ./squad_v2/dev-v2.0.json --per_gpu_train_batch_size 12 --per_gpu_eval_batch_size 8 --learning_rate 3e-5 --num_train_epochs 3.0 --max_seq_length 384 --doc_stride 128 --output_dir ./output_with_negative --save_steps 5000 --threads 4


## Pretrained model
You can download the pretrained weights from the [link](https://huggingface.co/bert-large-uncased-whole-word-masking/tree/main)
You can also download the pretrained weights from the [link](https://huggingface.co/bert-base-uncased)

## Special code
```python
# Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

# Save model checkpoint
if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)
```