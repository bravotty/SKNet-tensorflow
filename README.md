# SKNet-tensorflow
![auc][aucsvg]

[aucsvg]: <https://img.shields.io/badge/SKNet-tensorflow-blueviolet.svg>
[auc]: https://github.com/bravotty/SKNet-tensorflow



Simple tensorflow implementation of [Selective Kernel Networks](<https://arxiv.org/pdf/1903.06586.pdf>)

If you want to see the ***original author's code***, pls refer to this github [link](<https://github.com/implus/SKNet>)

***Version 1.0*** : SKNet block without groups and BN params. 

***Version 1.1***  : Set -> BN params. and remove -> fc layer BN ops.

* ***Version 1.2***  will be ***coming soon***.



## Requirements :

* Python >= 3.6
* Tensorflow >= 1.9.0
* Tflearn >= 0.3.2



## SKNet Block Structure :

![Selective Kernel Convolution](F:\githubRemote\SKNet-tensorflow\img\img1.png)

## Usage : 

```
import SKNet
import tensorflow.contrib.slim as slim

...
conv1 = slim.conv2d(inputs, 64, [3, 3], scope='conv1')
conv2 = SKNet(conv1, 3, 2, is_training=True)
conv3 = slim.conv2d(inputs, 3, [3, 3],  scope='out')
...

training code
```



## Be Careful ! 

At training stage, you may calculate the moving_mean and moving_var.

```
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
# Ensures that we execute the update_ops before performing the train_step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```

* Learn More :
  * Github Issue : [Easy to use batch norm layer](<https://github.com/tensorflow/tensorflow/issues/1122>)
  * Blog Tips : [tf-batch_normalization](<http://ruishu.io/2016/12/27/batchnorm/>)



## Contact :

Any improvement or bug-fixing is welcome. 

Create a [pull request](<https://github.com/bravotty/SKNet-tensorflow/pulls>) or [issue](<https://github.com/bravotty/SKNet-tensorflow/issues>) when you are done.



## License:

[The MIT License](<https://github.com/bravotty/SKNet-tensorflow/blob/master/LICENSE>)