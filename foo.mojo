from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort

fn echo(py_obj: PythonObject) raises -> PythonObject:
    var n = Int(py_obj)
    return n

@export
fn PyInit_bar() -> PythonObject:
    try:
        var m = PythonModuleBuilder("my lovely python binding!")
        m.def_function[echo]("my_mojo_echo")
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))
        return PythonObject()
