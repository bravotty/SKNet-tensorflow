# SKNet-tensorflow
Simple tensorflow implementation of [Selective Kernel Networks](<https://arxiv.org/pdf/1903.06586.pdf>)

If you want to see the ***original author's code***, pls refer to this github [link](<https://github.com/implus/SKNet>)

***Version 1.0*** : SKNet block without groups and BN params - ***coming soon***.



## Usage : 

```
import  SKNet
import tensorflow.contrib.slim as slim

...
conv1 = slim.conv2d(inputs, 64, [3, 3], scope='conv1')
conv2 = SKNet(conv1, 3, 2)
conv3 = slim.conv2d(inputs, 3, [3, 3],  scope='out')
...

training code
```

