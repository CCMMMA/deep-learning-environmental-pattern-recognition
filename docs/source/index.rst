Welcome to Deep Learning Environmental Pattern Recognition's documentation!
===========================================================================

A coupled numerical forecast and deep learning prediction model for environmental pattern recognition

**Introduction**

In the last decades, the impact of the global changes on human life became overwhelming.

The availability of computational resources and reliable numerical weather and marine models made crucial the production and the dissemination of accurate forecasts.

Nevertheless, intense weather events could be not detected by numerical weather and marine models due to diverse and different reasons as, but not limited to, spatial grid-scale, time scale, local behavior due to unmodeled domain characterization, and so on.

Traditionally those sources of uncertain forecasts were mitigated using data assimilation. While those techniques are feasible for synoptic modeling, they often can't be applied at a local scale due to the need for a finely distributed data acquisition network.

For this reason, the AI-augmented numerical prediction paradigm is fostering a promising set of machine learning and deep neural network-based solutions devoted to improving the weather and marine forecasts at a fraction of the current computational costs.

This repository is intended as practical support for the reproducibility of experiments described in scientific papers and journal articles.

Moreover, the new generation of students and scholars can be inspired in finding a new and better solution.

**Dataset**

The dataset available here are generated as environmental modeling simulation and forecasts by the Center for Marine and atmosphere Monitoring and Modeling (CMMMA, meteo@uniparthenope, http://meteo.uniparthenope.it) run by the University of Naples "Parthenope" (http://www.uniparthenope.it) and then professionally labeled or automatically labeled.

Each dataset is provided by a detailed description of the data format, of the labels and the use cases for algorithm testing and comparision.

Weather dataset are labeled using the Weather Labeling Web Application (https://github.com/CCMMMA/Weather-Labeling-Web-Application).

**Software**

**References**

* Montella, R., Di Luccio, D., Ciaramella, A., & Foster, I. (2019, October). StormSeeker: A Machine-Learning-Based Mediterranean Storm Tracer. In International Conference on Internet and Distributed Computing Systems (pp. 444-456). Springer, Cham.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   clustering
   models
   utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
