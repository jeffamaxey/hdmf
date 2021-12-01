import json
import os

from hdmf import Container
from hdmf.backends.hdf5.h5tools import HDF5IO
from hdmf.backends import SidecarValidationError
from hdmf.build import BuildManager, TypeMap, ObjectMapper
from hdmf.spec import AttributeSpec, DatasetSpec, GroupSpec, SpecCatalog, SpecNamespace, NamespaceCatalog
from hdmf.testing import TestCase
from hdmf.utils import getargs, docval


class TestBasic(TestCase):

    def setUp(self):
        self.h5_path = "./tests/unit/io_tests/test_sidecar.h5"
        foo2 = Foo('sub_foo', [-1, -2, -3], 'OLD', [-17])
        foo1 = Foo('foo1', [1, 2, 3], 'old', [17], foo2)
        with HDF5IO(self.h5_path, manager=_get_manager(), mode='w') as io:
            io.write(foo1)

        version2_0_0 = {
            "label": "2.0.0",
            "description": "change attr1 from 'old' to 'my experiment' and my_data from [1, 2, 3] to [4, 5]",
            "datetime": "2020-10-29T19:15:15.789Z",
            "agent": "John Doe",
            "changes": [
                {
                    "object_id": foo1.object_id,
                    "relative_path": "attr1",
                    "value": "my experiment"
                },
                {
                    "object_id": foo1.object_id,
                    "relative_path": "my_data",
                    "value": [4, 5],
                    "dtype": "int32"
                }
            ]
        }

        version3_0_0 = {
            "label": "3.0.0",
            "description": ("change sub_foo/my_data from [-1, -2, -3] to [[0]], delete my_data/attr2, and change "
                            "dtype of my_data"),
            "datetime": "2021-11-30T20:16:16.790Z",
            "agent": "Jane Doe",
            "changes": [
                {
                    "object_id": foo2.object_id,
                    "relative_path": "my_data",
                    "value": [[0]]
                },
                {
                    "object_id": foo1.object_id,
                    "relative_path": "my_data/attr2",
                    "value": None  # will be encoded on disk as null
                },
                {
                    "object_id": foo1.object_id,
                    "relative_path": "my_data",
                    "value": [6, 7],
                    "dtype": "int8"
                },
            ]
        }

        version3_0_1 = {
            "label": "3.0.1",
            "description": "change my_data from [4, 5] to [6, 7]",
            "datetime": "2021-11-30T20:17:16.790Z",
            "agent": "Jane Doe",
            "changes": [
                {
                    "object_id": foo1.object_id,
                    "relative_path": "my_data",
                    "value": [6, 7],
                },
            ]
        }

        sidecar = dict()
        sidecar["versions"] = [version2_0_0, version3_0_0, version3_0_1]
        sidecar["schema_version"] = "0.1.0"

        self.json_path = "./tests/unit/io_tests/test_sidecar.json"
        with open(self.json_path, 'w') as outfile:
            json.dump(sidecar, outfile, indent=4)

    def tearDown(self):
        if os.path.exists(self.h5_path):
            os.remove(self.h5_path)
        if os.path.exists(self.json_path):
            os.remove(self.json_path)

    def test_update_builder(self):
        with HDF5IO(self.h5_path, 'r', manager=_get_manager()) as io:
            foo1 = io.read()
        assert foo1.attr1 == "my experiment"
        assert foo1.my_data == [6, 7]  # TODO test dtype
        assert foo1.sub_foo.my_data == [[0]]
        assert foo1.attr2 is None


class TestFailValidation(TestCase):

    def setUp(self):
        self.h5_path = "./tests/unit/io_tests/test_sidecar_fail.h5"
        foo2 = Foo('sub_foo', [-1, -2, -3], 'OLD', [-17])
        foo1 = Foo('foo1', [1, 2, 3], 'old', [17], foo2)
        with HDF5IO(self.h5_path, manager=_get_manager(), mode='w') as io:
            io.write(foo1)

        sidecar = dict()
        sidecar["versions"] = []

        self.json_path = "./tests/unit/io_tests/test_sidecar_fail.json"
        with open(self.json_path, 'w') as outfile:
            json.dump(sidecar, outfile, indent=4)

    def tearDown(self):
        if os.path.exists(self.h5_path):
            os.remove(self.h5_path)
        if os.path.exists(self.json_path):
            os.remove(self.json_path)

    def test_simple(self):
        with HDF5IO(self.h5_path, 'r', manager=_get_manager()) as io:
            with self.assertRaises(SidecarValidationError):
                io.read()


class Foo(Container):

    @docval({'name': 'name', 'type': str, 'doc': 'the name of this Foo'},
            {'name': 'my_data', 'type': ('array_data', 'data'), 'doc': 'a 1-D integer dataset'},
            {'name': 'attr1', 'type': str, 'doc': 'a string attribute'},
            {'name': 'attr2', 'type': ('array_data', 'data'), 'doc': 'a 1-D integer attribute', 'default': None},
            {'name': 'sub_foo', 'type': 'Foo', 'doc': 'a child Foo', 'default': None})
    def __init__(self, **kwargs):
        name, my_data, attr1, attr2, sub_foo = getargs('name', 'my_data', 'attr1', 'attr2', 'sub_foo', kwargs)
        super().__init__(name=name)
        self.__data = my_data
        self.__attr1 = attr1
        self.__attr2 = attr2
        self.__sub_foo = sub_foo
        if sub_foo is not None:
            assert sub_foo.name == 'sub_foo'  # on read mapping will not work otherwise
            self.__sub_foo.parent = self

    @property
    def my_data(self):
        return self.__data

    @property
    def attr1(self):
        return self.__attr1

    @property
    def attr2(self):
        return self.__attr2

    @property
    def sub_foo(self):
        return self.__sub_foo


def _get_manager():
    foo_spec = GroupSpec(
        doc='A test group specification with a data type',
        data_type_def='Foo',
        groups=[
            GroupSpec(
                doc='a child Foo',
                data_type_inc='Foo',
                name='sub_foo',
                quantity='?',
            )
        ],
        datasets=[
            DatasetSpec(
                doc='a 1-D integer dataset',
                dtype='int',
                name='my_data',
                shape=[None, ],
                attributes=[
                    AttributeSpec(
                        name='attr2',
                        doc='a 1-D integer attribute',
                        dtype='int',
                        shape=[None, ],
                        required=False
                    )
                ]
            )
        ],
        attributes=[
            AttributeSpec(name='attr1', doc='a string attribute', dtype='text'),
        ]
    )

    class FooMapper(ObjectMapper):
        """Remap 'attr2' attribute on Foo container to 'my_data' dataset spec > 'attr2' attribute spec."""
        def __init__(self, spec):
            super().__init__(spec)
            my_data_spec = spec.get_dataset('my_data')
            self.map_spec('attr2', my_data_spec.get_attribute('attr2'))

    spec_catalog = SpecCatalog()
    spec_catalog.register_spec(foo_spec, 'test.yaml')
    namespace_name = 'test_core'
    namespace = SpecNamespace(
        doc='a test namespace',
        name=namespace_name,
        schema=[{'source': 'test.yaml'}],
        version='0.1.0',
        catalog=spec_catalog
    )
    namespace_catalog = NamespaceCatalog()
    namespace_catalog.add_namespace(namespace_name, namespace)
    type_map = TypeMap(namespace_catalog)
    type_map.register_container_type(namespace_name, 'Foo', Foo)
    type_map.register_map(Foo, FooMapper)
    manager = BuildManager(type_map)
    return manager
