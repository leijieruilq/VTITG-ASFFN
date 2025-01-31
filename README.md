# VTITG-ASFFN


## running programme

### Single-process experiment: running exp.py

> >Running style


> >(1) Setting up the experimental task environment: you can do a manual setup of parser.add_argument in exp.py

> >1.1 "model_name":"vtitg-asffn"

> >1.2 "dataset_name": The corresponding "help" in exp.py selects the dataset.

> >1.3 "inp_len": 96

> >1.4 "pred_len": 96/192/336/720

> >(2) Run it directly from the command line：nohup python -u exp.py > train.log 2&>1 &

> >(3) No pre-setting, run directly from the command line：

> > for example：nohup python -u exp.py --note "vtitg-asffn-traffic-96" --model_name "vtitg-asffn" --dataset_name "traffic" --inp_len 96 --pred_len 96 > train.log 2>&1 &

> > The results are in the corresponding train.log file.

> > We provide (Weather) MAV and (Traffic) MEV results log as example. You can see Model Settings in the logs, like this:

> > Model Setting:
    
    name: "vtitg-asffn";      adp_dim: "10";      layers: "1";      order: "2";      dilation: "1";  
    kernel_size: "3";      dropout: "0.5";      share: "False";      use_update: "False";  
    use_guide: "True";      use_mav: "True";      c_date: "5";      n_nodes: "1";      c_in: "21";  
    c_out: "21";      device: "cuda:0";      inp_len: "96";      pred_len: "720";  
