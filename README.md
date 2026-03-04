# （MLJ 2026）Collaborative Multivariate Time Series Forecasting via Variable-Tailored Inter-Temporal Graph and Adaptive-Smooth Frequency Fusion

## Introduction

Frequency domain learning and accurate multivariate dependencies are crucial for driving multivariate time series forecasting applications in real world. However, the existing progress remains limited. First, the multivariate variables can be divided into “Multi-Attribute” and “Multi-Entity” types, which necessitate considerations for tailored correlation modeling and dynamic dependencies capturing. Second, the inherent non-stationary of time series conflicts with the stationary assumptions of frequency analysis, and the temporal globality of Fourier basis function tends to neglect local information. To address these challenges, we propose the Variable-Tailored Inter-Temporal Graph and Adaptive-Smooth Frequency Fusion Network (VTITG-ASFFN), which first adaptively stabilizes time series through mask learning and realizes local–global collaborative learning by frequency components mining and fusion, then a notable innovation is a tailored inter-temporal graph for “Multi-Attribute” and “Multi-Entity” correlation scenarios, which effectively interacts with input series via Variable-Tailored Adaptive Graph and Channels-Time Graph, learning “dynamic spatial-temporal dependencies” in temporal context, enabling high-fidelity evolution of “Multi-Attribute” and dynamic understanding of correlations among “Multi-Entity”. Evaluations on 8 real-world datasets demonstrate the superiority of VTITG-ASFFN in forecasting, efficiency and universality over SoTA benchmarks. 

## VTITG-ASFFN Architecture

<img width="8878" height="4811" alt="image" src="https://github.com/user-attachments/assets/6c3ce434-4282-485f-8c8d-70f463d3ded7" />

<img width="9895" height="4369" alt="image" src="https://github.com/user-attachments/assets/6b16f6c4-d3b4-46cd-a4e0-b8b210cb0cb2" />

## Averaged Results

<img width="762" height="197" alt="{929323C7-EC6F-4550-87E0-46960F9A225F}" src="https://github.com/user-attachments/assets/47b5c934-7bd6-461b-846d-f5f0fb5ca380" />

<img width="626" height="182" alt="{0FCC3A4F-635E-44C2-B70D-8FA5F10DC35F}" src="https://github.com/user-attachments/assets/2e4d5a40-4000-4d1a-8943-439169ceee13" />

## Running Programme

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
