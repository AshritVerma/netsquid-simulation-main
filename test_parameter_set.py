import unittest
from netsquid_simulationtools.parameter_set import Parameter, ParameterSet, linear_improvement_fn, rootbased_improvement_fn


class _SpecificParameterSet(ParameterSet):

    _REQUIRED_PARAMETERS = [
        Parameter(name="A",
                  type=float,
                  units="some_units",
                  perfect_value=3.),
        Parameter(name="b",
                  type=int,
                  units=None,
                  perfect_value=42)]


class _SpecificParameterSet2019TooFew(_SpecificParameterSet):
    A = 5.


class _SpecificParameterSet2019TypeError(_SpecificParameterSet):
    A = 5
    b = 9


class _SpecificParameterSet2019(_SpecificParameterSet):
    A = 3.
    b = 6


class _NotSpecifiedParameterSet(_SpecificParameterSet):
    A = 3.
    b = ParameterSet._NOT_SPECIFIED


class TestParameterSet(unittest.TestCase):

    def test_initialization_incomplete(self):
        """
        Test whether the parameter set has
        too few parameters defined indeed
        raises an error.
        """
        with self.assertRaises(AttributeError):
            _SpecificParameterSet2019TooFew()

    def test_initialization_wrong_type(self):
        with self.assertRaises(TypeError):
            _SpecificParameterSet2019TypeError()

    def test_initialization_complete(self):
        _SpecificParameterSet2019()
        _NotSpecifiedParameterSet()

    def test_to_dict(self):
        sut_output = _SpecificParameterSet2019().to_dict()
        expected_output = {"A": 3., "b": 6}
        self.assertEqual(sut_output, expected_output)

    def test_to_improved_dict(self):

        def custom_improvement_fn(parameter, current_value, constant):
            return parameter.perfect_value + current_value + constant

        param_dict = {"A": 1., "b": 2}

        # improve only a single parameter
        param_improvement_dict = {"A": 9}
        sut_output = \
            _SpecificParameterSet.to_improved_dict(
                param_dict=param_dict,
                param_improvement_dict=param_improvement_dict,
                improvement_fn=custom_improvement_fn)
        expected_output = {"A": 1. + 3. + 9, "b": 2}
        self.assertEqual(sut_output, expected_output)

        # improve all parameters
        param_improvement_dict = {"A": 9, "b": -4}
        sut_output = \
            _SpecificParameterSet.to_improved_dict(
                param_dict=param_dict,
                param_improvement_dict=param_improvement_dict,
                improvement_fn=custom_improvement_fn)
        expected_output = {"A": 1. + 3. + 9, "b": 2 + -4 + 42}
        self.assertEqual(sut_output, expected_output)

    def test_parameter_names(self):
        sut_output = _SpecificParameterSet2019().parameter_names()
        expected_output = ["A", "b"]
        self.assertEqual(sut_output, expected_output)

    def test_linear_improvement_fn(self):
        my_parameter = Parameter(name="myparam",
                                 units=None,
                                 perfect_value=10.0,
                                 type=float)
        sut_output = \
            linear_improvement_fn(parameter=my_parameter,
                                  current_value=2.,
                                  scalar=0.8)
        expected_output = 8.4
        self.assertEqual(sut_output, expected_output)

    def test_rootbased_improvement_fn(self):
        my_parameter = Parameter(name="myparam",
                                 units=None,
                                 perfect_value=0.5,
                                 type=float,
                                 convert_to_prob_fn=lambda x: 1 - x,
                                 convert_from_prob_fn=lambda x: 1 - x)
        sut_output = rootbased_improvement_fn(
            parameter=my_parameter,
            current_value=1. - 0.027,  # 0.027 = 27 * 10^(-3) = (3 * 10^(-1))^3
            factor=3)
        expected_output = 1. - 0.3
        self.assertEqual(sut_output, expected_output)


if __name__ == "__main__":
    unittest.main()
