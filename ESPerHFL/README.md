For running ESPerHFL with mnist and lenet:
```
python3 ESPerHFL 
--dataset mnist 
--model lenet 
--num_clients 50 
--num_edges 10 
--num_local_update 60 
--num_edge_aggregation 1 
--num_communication 120
--batch_size 20 
--lr 0.01
--lr_decay 0.995
--lr_decay_epoch 1
--momentum 0
--weight_decay 0
```