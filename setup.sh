mkdir save
curl https://ptemplin-pytorch-models.s3.amazonaws.com/bert/bert_base_75f1/config.json > save/config.json
curl https://ptemplin-pytorch-models.s3.amazonaws.com/bert/bert_base_75f1/optimizer.pt > save/optimizer.pt
curl https://ptemplin-pytorch-models.s3.amazonaws.com/bert/bert_base_75f1/pytorch_model.bin > save/pytorch_model.bin
curl https://ptemplin-pytorch-models.s3.amazonaws.com/bert/bert_base_75f1/scheduler.pt > save/scheduler.pt
curl https://ptemplin-pytorch-models.s3.amazonaws.com/bert/bert_base_75f1/special_tokens_map.json > save/special_tokens_map.json
curl https://ptemplin-pytorch-models.s3.amazonaws.com/bert/bert_base_75f1/tokenizer_config.json > save/tokenizer_config.json
curl https://ptemplin-pytorch-models.s3.amazonaws.com/bert/bert_base_75f1/trainer_state.json > save/trainer_state.json
curl https://ptemplin-pytorch-models.s3.amazonaws.com/bert/bert_base_75f1/vocab.txt > save/vocab.txt
