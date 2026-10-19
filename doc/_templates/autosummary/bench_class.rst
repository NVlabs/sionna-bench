{{ name | escape | underline }}

{% if fullname in nn_asset_api_classes %}
.. note::

   - Estimator constructor defaults mirror the settings used during benchmarking.
   - ``{{ nn_asset_api_members[fullname][0] }}()`` defaults mirror the settings used to generate training data.
   - ``{{ nn_asset_api_members[fullname][1] }}()`` defaults mirror the settings used to train the model.

{% endif %}

{% if fullname in statistical_asset_api_classes %}
.. note::

   - Estimator constructor defaults mirror the settings used during benchmarking.
   - ``generate_fitting_data()`` defaults mirror the settings used to generate fitting data.
   - ``fit_model()`` defaults mirror the settings used to fit the model.

{% endif %}

{% if fullname in covariance_asset_api_classes %}
.. note::

   - Estimator constructor defaults mirror the settings used during benchmarking.
   - ``estimate_covariance()`` defaults mirror the settings used to estimate the covariance.

{% endif %}

{% if fullname in tuning_api_classes and fullname not in nn_asset_api_classes and fullname not in statistical_asset_api_classes and fullname not in covariance_asset_api_classes %}
.. note::

   - Estimator constructor defaults mirror the settings used during benchmarking.
{% endif %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
{% if bench_integrated and fullname.startswith("sionna.phy.") %}   :no-index:
{% endif %}

{% if fullname in asset_api_modules %}
{% set asset_module = asset_api_modules[fullname] %}
{% else %}
{% set asset_module = module ~ ".asset" %}
{% endif %}

{% if fullname in nn_asset_api_classes %}
Neural Network Training
-----------------------

.. automodule:: {{ asset_module }}
   :no-index:

{% for member in nn_asset_api_members[fullname] %}
.. autofunction:: {{ asset_module }}.{{ member }}
{% endfor %}
{% endif %}

{% if fullname in statistical_asset_api_classes %}
Statistical Fitting
-------------------

.. automodule:: {{ asset_module }}
   :members:
   :exclude-members: load_model, load_fitting_data, load_covariance
   :member-order: bysource
{% if fullname in shared_auxiliary_api_classes %}   :no-index:
{% endif %}
{% endif %}

{% if fullname in covariance_asset_api_classes %}
Covariance Estimation
---------------------

.. automodule:: {{ asset_module }}
   :members:
   :exclude-members: load_covariance
   :member-order: bysource
{% endif %}


{% if fullname in tuning_api_classes %}
Hyperparameter Tuning
---------------------

.. automodule:: {{ module }}.tuning
   :members:
{% if fullname in shared_auxiliary_api_classes %}   :no-index:
{% endif %}
{% endif %}
