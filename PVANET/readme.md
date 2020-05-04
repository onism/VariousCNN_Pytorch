PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection
by Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, and Jiajun Liang
Link to the paper: https://arxiv.org/pdf/1608.08021.pdf


***The key design principle is 'less channels with more layers' ***

1. Crelu is applied to the early stage of CNNs to reduce the computations
2. Inception is applied to the remaining of feature generation sub-networks
3. Adopted the idea of multi-scale representation that combines several intermediate outputs so that multiple levels of details and nonlinearities can be considered simulataneously.

CRelu

x -> conv(x) -> negation --concatenation --->scale/shift -->relu
        |__________________|


Inception
replace 5 * 5 by two 3 * 3
      _____________________conv(1 \times 1)_______________________________|
     |                                                                    |  
     |                                                                    |  
x ---------------conv(1\times 1) ----- conv(3\times 3) -------------------|---concat---conv( 1 \times 1)
     |                                                                    |  
     |________conv(1 \times  1)----- conv(3\times 3)----- conv(3\times 3)-|


 reducing feature map size by half
      _____________________conv(1 \times 1, stide=2)_______________________________ |
     |                                                                              |  
     |                                                                              |  
x ---------------conv(1\times 1, stride=2) ----- conv(3\times 3) -------------------|---concat---conv( 1 \times 1)
     |                                                                              |  
     |________conv(1 \times  1, stride=2)----- conv(3\times 3)----- conv(3\times 3)-|
     |                                                                              |
     |________pool(3 \times 3, stride=2)----- conv(1\times 1)------------------------   

Multiscale representation and its combination are proven to be effective in many Deep Learning tasks. Combining fine grained details with highly abstracted information in feature extraction layer helps the following region proposal network and classification network to detect object of different scales.
They combine the
1) Last layer
2) Two intermediate layers whose scales are 2x and 4x of the last layer, respectively.