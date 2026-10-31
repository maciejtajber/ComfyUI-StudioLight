"""Regression coverage for saved-workflow migration; no ComfyUI server needed."""
import copy
import unittest
from tools.migrate_workflows import WIDGETS, default_of, migrate_values, fix_api, fix_graph

class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.current = [default_of(k) for k in WIDGETS]
        self.legacy = [self.current[0], 'Portrait', self.current[1], self.current[2]]
        for light in ('key', 'fill', 'rim'):
            self.legacy += [default_of(light + '_' + field) for field in ('enabled', 'azimuth', 'elevation', 'intensity', 'kelvin', 'modifier')]

    def test_current_prefixes_preserve_every_existing_value(self):
        for count in range(26, 32):
            with self.subTest(count=count):
                result, _ = migrate_values(self.current[:count])
                self.assertEqual(result, self.current)

    def test_legacy_layouts_preserve_lamp_settings(self):
        for values in (self.legacy, self.legacy + [0.4, 1.0, 1.0, 'A_NORM']):
            result, note = migrate_values(values)
            self.assertTrue(note.startswith('legacy-'))
            by_name = dict(zip(WIDGETS, result))
            for light in ('key', 'fill', 'rim'):
                self.assertEqual(by_name[light + '_distance'], 1.8)
                for field in ('enabled', 'azimuth', 'elevation', 'intensity', 'kelvin', 'modifier'):
                    self.assertEqual(by_name[light + '_' + field], default_of(light + '_' + field))
            self.assertEqual(by_name['normal_boost'], 1.0)
            self.assertEqual(by_name['ao_anchoring'], 0.0)

    def test_invalid_layouts_are_refused(self):
        for index, value in [(1, False), (8, 0.35), (9, 4800), (4, float('nan'))]:
            values = self.current.copy()
            values[index] = value
            self.assertIsNone(migrate_values(values)[0])
        self.assertIsNone(migrate_values(self.legacy[:21])[0])

    def test_api_accepts_all_current_outputs(self):
        for slot in range(5):
            graph = {'1': {'class_type': 'StudioLightRig', 'inputs': dict(zip(WIDGETS, self.current))},
                     '2': {'class_type': 'StudioPhotometricComposite', 'inputs': {'specular_map': ['1', slot]}}}
            before = copy.deepcopy(graph)
            self.assertEqual(fix_api(graph)[0], 'ok')
            self.assertEqual(graph, before)

    def test_api_refuses_invalid_output_without_mutation(self):
        for slot in (-1, 5, 99, '3', True):
            graph = {'1': {'class_type': 'StudioLightRig', 'inputs': {}},
                     '2': {'class_type': 'StudioPhotometricComposite', 'inputs': {'specular_map': ['1', slot]}}}
            before = copy.deepcopy(graph)
            self.assertEqual(fix_api(graph)[0], 'obsolete')
            self.assertEqual(graph, before)

    def test_graph_appends_defaults_without_shifting_settings(self):
        graph = {'nodes': [{'id': 1, 'type': 'StudioLightRig', 'widgets_values': self.current[:26]}], 'links': []}
        self.assertEqual(fix_graph(graph)[0], 'migrated')
        self.assertEqual(graph['nodes'][0]['widgets_values'], self.current)
        self.assertEqual(fix_graph(graph)[0], 'ok')

if __name__ == '__main__':
    unittest.main()
