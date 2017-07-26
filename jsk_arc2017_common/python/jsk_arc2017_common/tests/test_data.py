from unittest import TestCase

from jsk_arc2017_common import data


class TestData(TestCase):

    def test_get_known_object_names(self):
        known_object_names = data.get_known_object_names()
        self.assertIsInstance(known_object_names, list)
        self.assertEqual(len(known_object_names), 40)
        for obj_name in known_object_names:
            self.assertIsInstance(obj_name, basestring)

    def test_get_label_names(self):
        label_names = data.get_label_names()
        self.assertIsInstance(label_names, list)
        self.assertEqual(label_names[0], '__background__')
        for label_names in label_names:
            self.assertIsInstance(label_names, basestring)

    def test_get_object_names(self):
        object_names = data.get_object_names()
        self.assertIsInstance(object_names, list)
        for obj_name in object_names:
            self.assertIsInstance(obj_name, basestring)

    def test_get_object_graspability(self):
        object_graspability = data.get_object_graspability()
        self.assertIsInstance(object_graspability, dict)

        object_names = data.get_object_names()
        for obj_name in object_names:
            self.assertTrue(obj_name in object_graspability)
            grasp_graspability = object_graspability[obj_name]
            self.assertIsInstance(grasp_graspability, dict)
            for grasp_style in ['suction', 'pinch']:
                self.assertTrue(grasp_style, grasp_graspability)
                graspability = grasp_graspability[grasp_style]
                self.assertIsInstance(graspability, int)
                self.assertGreaterEqual(graspability, 1)
                self.assertLessEqual(graspability, 3)

    def test_get_object_weights(self):
        object_weights = data.get_object_weights()
        self.assertIsInstance(object_weights, dict)

        object_names = data.get_object_names()
        for obj_name in object_names:
            self.assertTrue(obj_name in object_weights)
            weight = object_weights[obj_name]
            self.assertIsInstance(weight, (int, float))
            self.assertGreaterEqual(weight, 0)
            self.assertLessEqual(weight, 1000)
