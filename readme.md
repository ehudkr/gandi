# GANDI: Generative Adversarial Networks for Detecting Irregularities

Harnessing the usually discarded GAN discriminator for the task of 
anomaly detection. 

## Motivation

*\[For more comprehensive explanation, see the pdf file in the paper 
folder]*  

Anomaly detection is hard. The possible number of different types of 
anomalies cannot be accounted for. Thus, in order to train an anomaly 
detector we must use a generative model, rather than a discriminative one.
  
We hypothesize the discriminator performance over time (during training)
as parabolic: it begins knowing nothing (randomly initialized) and ends 
up confused (being the generator winning the competition). However, in 
between these two ends, we know it learn some  meaningful representation 
of the problem of identifying true data from false one; otherwise it 
wouldn't be able to contribute to the performance improvement of its 
generator foe - which we know to improve for sure (we can test that).  
at some point in our training, the discriminator outputs values near 1 
when encountered with true data and near 0 for data impersonated to be 
true data (adversarial generated synthetic data). If we could find the
sweet-spot where the discriminator acts it's best (i.e. discriminates
between true and fake with good identifiable margin), we could pause the
training and export (or rather metamorphose) our discriminator as an anomaly 
detector - feeding it real data will output near constant output (~1)
and feeding it anomalies (data that was not "real" during training) will
output some other value.  

After the discriminator metamorphoses as an anomaly detector, we can test
is as if it was a binary classification problem: see how it performs when
encountering two sets of data - true data (labeled 1, like during training)
and anomaly data (labeled 0). We can then apply several metrics on the 
resulting confusion matrix, mainly the area under the curve (AUC).

### Results

The generator converging to the true data (PDF and CDF):  
![pdf]() ![cdf]()

The discriminator performance as anomaly detector during training:
![roc curve of GAN anomaly detector for growing anomalies]()

## Code

### Description of Files:

[//]: # (Convert the Bold type face to links after publishing in GitHub)
 
* **PlayGround** - Main file running the project.
* **RunParams** - Static file with setting declarations (the NN architecture,
                  training and losses, training parameters, what tests to
                  perform and how often, etc.)
* **Distributions** - Class depicting different distributions being used in
                      the demonstration. E.g. wrapping SciPy's Gaussian 
                      distribution.
* **DiscriminatorNN** - Discriminator class and different discriminator's
                        neural architectures.
* **GeneratorNN** - Generator class and different generator's neural 
                    architectures.
* **NNbuilds** - Helper for defining linear layers and optimizers for 
                 TensorFlow networks. Shared to both discriminator and
                 generator.
* **Tracker** - An object tracking the progress of training process. It
                is called every iteration of training and decides id test
                should be conducted and registered and if there suppose to
                be any logging of current models parameters.
* **MetricsG** - Module of different goodness-of-fit tests to be conducted
                 on the generator during training process.
* **MetricsD** - Object for testing the discriminator (as an anomaly
                 detector during training).
* **GAN** - Class for training the GAN model.
* **Plots** - Plotting class generating different plots based on the resulting
              statistics of the generator, discriminator (anomaly detector),
              and the neural network training progress.

### Prerequisites

* [TensorFlow](https://www.tensorflow.org/) (Ver 1.n)
* [pandas](http://pandas.pydata.org/)
* [NumPy](http://www.numpy.org/) + [SciPy](https://www.scipy.org/)
* [StatsModels](http://www.statsmodels.org/stable/index.html)
* [Seaborn](https://seaborn.pydata.org/)

### Workspace Structure

Running *PlayGround.py* creates many files, each with a unique name 
(signature) for every model-configuration run:
* Plots (.png files)
* TensorFlow's Saver checkpoints
* Logs (text based)
* Tensorboard files
* results (pickle file of plots objects (figure), and raw measurements 
          (DataFrame))

Please make sure that before running, you have the following workspace
structure of subdirectories:
```
gandi-
     |-GAN.py
     |-PlayGround.py
     |-... [All the rest of the files described above]
     |-log_dir-
              |-plots
              |-checkpoints
              |-logs
              |-tensorboard
              |-results
```

Each model configuration will create a unique subdirectory (or file) under
this structure using the time of start, random state nd configuration
setting number.  
For example:  
`GAN_2017-05-23_13-47-50_966105-765644_9`

### Running

1. There's a shebang header in the *PlayGround* so you can simply  
    ```
    $ cd SOME_PATH/gandi
    $ PlayGround.py
    ```
    it in the terminal.  
2. In order to change settings and configuration you should edit the 
   variables in *RunParams.py*.
3. In order to expand the model just add another neural net architecture
   in *GeneratorNN* and\or *DiscriminatorNN*:  
   `def architecture_n(net_input, ...)`  
   and update the factory dictionary `_arch_types`:  
   ```
   _arch_types = {"architecture_1": architecture_1,
                  ...
                  "architecture_n": architecture_n}
   ```  
   and then add it to the *RunParam.py* `train_params` dictionary:  
   `{"d_arch_num": n, "g_arch_num": m}`

## Authors

* **Ehud Karavani** - *solo contributor* - [GitHub](https://github.com/ehudkr)
* **Dr. Matan Gavish** - *academic adviser* - [personal website](https://web.stanford.edu/~gavish/) 

## License

Please contact before citing, using or if willing to expand this work.  
(Only to resolve credits for this not-formally-published work :blush:) 

## Acknowledgments

* The Python and scientific-Python development community.
* [JetBrains](https://www.jetbrains.com/) for free 
  [PyCharm](https://www.jetbrains.com/pycharm/) license for students. 

