//! Private bytes-only access to the shared adapter. No Python syntax is lowered here.

use egglog_experimental::protobuf::NativeExtension;
use egglog_experimental::protobuf::{Engine, TransportError};
use prost::Message;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Generation-time export from the actual registrar, not a parallel catalog.
#[pyfunction]
pub(crate) fn _export_pyobject_catalog(py: Python<'_>) -> PyResult<Bound<'_, PyBytes>> {
    let mut graph = egglog_experimental::new_experimental_egraph();
    crate::py_object_sort::PyObjectExtension::default()
        .initialize(&mut graph)
        .map_err(PyValueError::new_err)?;
    let catalog = graph
        .type_info()
        .builtin_catalog()
        .map_err(PyValueError::new_err)?;
    eprintln!(
        "undescribed primitives: {:?}; families: {:?}; family primitives: {:?}; sorts: {:?}",
        catalog.undescribed_primitives,
        catalog.undescribed_families,
        catalog.undescribed_family_primitives,
        catalog.undescribed_sorts
    );
    Ok(PyBytes::new(py, &catalog.definitions.encode_to_vec()))
}

#[pyclass(name = "_ProtoEngine")]
#[derive(Default)]
pub struct ProtoEngine {
    engine: Engine,
}

impl ProtoEngine {
    fn request<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
        operation: fn(&mut Engine, &[u8]) -> Result<Vec<u8>, TransportError>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        // Own the input while the GIL is released; only encoded bytes cross
        // this boundary in either direction, including structured run errors.
        let bytes = request.as_bytes().to_vec();
        let response = py
            .detach(|| operation(&mut self.engine, &bytes))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PyBytes::new(py, &response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::py_object_sort::{PyObjectExtension, PyPickledValue, dump, load};
    use egglog_experimental::proto as pb;
    use egglog_experimental::protobuf::NativeExtension;
    use prost::Message;
    use pyo3::types::PyDict;
    use std::sync::Arc;

    #[test]
    fn provider_catalog_and_inert_string_bytes() {
        Python::initialize();
        let extension = Arc::new(PyObjectExtension::default());
        let mut graph = egglog_experimental::new_experimental_egraph();
        extension.initialize(&mut graph).unwrap();
        let catalog = graph.type_info().builtin_catalog().unwrap();
        assert!(
            !catalog
                .undescribed_sorts
                .iter()
                .any(|name| name == "PyObject")
        );
        assert!(
            !catalog
                .undescribed_primitives
                .iter()
                .any(|name| name.starts_with("py-"))
        );
        let mut keys = catalog
            .definitions
            .declarations
            .iter()
            .filter_map(|d| match &d.kind {
                Some(pb::declaration::Kind::HostPrimitive(p))
                    if p.name.starts_with("egglog.python.pyobject.") =>
                {
                    Some(p.name.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            keys,
            [
                "call",
                "call_extended",
                "dict",
                "dict_update",
                "eval",
                "exec",
                "from_int",
                "from_pickle",
                "from_string",
                "to_bool",
                "to_string"
            ]
            .map(|name| format!("egglog.python.pyobject.{name}"))
        );
        let family = catalog
            .definitions
            .declarations
            .iter()
            .find_map(|declaration| match &declaration.kind {
                Some(pb::declaration::Kind::HostSortFamily(family))
                    if family.name == "PyObject" =>
                {
                    Some(family)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(family.arity, 0);
        assert_eq!(
            family
                .bindings
                .as_ref()
                .unwrap()
                .python
                .as_ref()
                .unwrap()
                .path,
            ["egglog", "builtins", "PyObject"]
        );
        assert!(family.bindings.as_ref().unwrap().rust.is_none());
        for (key, alias, inputs, tail, output) in [
            (
                "from_pickle",
                "py-object",
                vec!["String"],
                vec![],
                "PyObject",
            ),
            (
                "eval",
                "py-eval",
                vec!["String", "PyObject", "PyObject"],
                vec![],
                "PyObject",
            ),
            (
                "exec",
                "py-exec",
                vec!["String", "PyObject", "PyObject"],
                vec![],
                "PyObject",
            ),
            ("call", "py-call", vec![], vec!["PyObject"], "PyObject"),
            ("dict", "py-dict", vec![], vec!["PyObject"], "PyObject"),
            (
                "dict_update",
                "py-dict-update",
                vec![],
                vec!["PyObject"],
                "PyObject",
            ),
            (
                "call_extended",
                "py-call-extended",
                vec!["PyObject", "PyObject", "PyObject"],
                vec![],
                "PyObject",
            ),
            (
                "to_string",
                "py-to-string",
                vec!["PyObject"],
                vec![],
                "String",
            ),
            ("to_bool", "py-to-bool", vec!["PyObject"], vec![], "bool"),
            (
                "from_string",
                "py-from-string",
                vec!["String"],
                vec![],
                "PyObject",
            ),
            ("from_int", "py-from-int", vec!["i64"], vec![], "PyObject"),
        ] {
            let key = format!("egglog.python.pyobject.{key}");
            let declaration = catalog.definitions.declarations.iter().find(|d| matches!(&d.kind, Some(pb::declaration::Kind::HostPrimitive(p)) if p.name == key)).unwrap();
            let Some(pb::declaration::Kind::HostPrimitive(primitive)) = &declaration.kind else {
                unreachable!()
            };
            let Some(pb::host_primitive::Typing::Signature(signature)) = &primitive.typing else {
                panic!("not a real closed signature")
            };
            assert!(signature.type_params.is_empty());
            let sort_name = |index: u32| {
                let Some(pb::sort::Kind::Family(family)) =
                    &catalog.definitions.sorts[index as usize].kind
                else {
                    panic!("not a host sort")
                };
                assert!(family.args.is_empty());
                family.name.as_str()
            };
            assert_eq!(
                signature
                    .inputs
                    .iter()
                    .map(|arg| sort_name(arg.sort))
                    .collect::<Vec<_>>(),
                inputs
            );
            assert_eq!(
                signature
                    .varargs
                    .iter()
                    .map(|arg| sort_name(arg.sort))
                    .collect::<Vec<_>>(),
                tail
            );
            assert_eq!(sort_name(signature.output.unwrap()), output);
            let bindings = declaration.bindings.as_ref().unwrap();
            assert_eq!(bindings.egglog.as_ref().unwrap().views[0].symbol, alias);
            if matches!(alias, "py-object" | "py-call" | "py-dict-update") {
                assert!(bindings.python.is_none());
            }
            // The alias and key refer to the same actual registration in every
            // native context. This resolves signatures; no callback is executed.
            let sorts = inputs
                .iter()
                .chain(&tail)
                .copied()
                .chain([output])
                .map(|name| graph.get_sort_by_name(name).unwrap().clone())
                .collect::<Vec<_>>();
            for context in [
                egglog_experimental::Context::Pure,
                egglog_experimental::Context::Read,
                egglog_experimental::Context::Write,
                egglog_experimental::Context::Full,
            ] {
                let mut resolve = |name: &str| {
                    egglog_experimental::ResolvedCall::from_resolution(
                        name,
                        &sorts,
                        graph.type_info(),
                        context,
                        &egglog::span!(),
                    )
                    .unwrap()
                };
                let (
                    egglog_experimental::ResolvedCall::Primitive(a),
                    egglog_experimental::ResolvedCall::Primitive(b),
                ) = (resolve(alias), resolve(&key))
                else {
                    panic!("not native primitives")
                };
                assert_eq!(a, b);
            }
        }
        let exported =
            Python::attach(|py| _export_pyobject_catalog(py).unwrap().as_bytes().to_vec());
        assert_eq!(exported, catalog.definitions.encode_to_vec());
        assert_eq!(
            pb::Program::decode(exported.as_slice())
                .unwrap()
                .encode_to_vec(),
            exported
        );
        assert_eq!(
            graph.type_info().builtin_catalog().unwrap().definitions,
            catalog.definitions
        );
        let payload = Python::attach(|py| dump("hello".into_pyobject(py).unwrap()).unwrap().0);
        let mut engine = Engine::with_extension(extension);
        let id = create(&mut engine);
        let catalog_result = run(&mut engine, id, catalog.definitions.clone());
        assert_eq!(catalog_result.error, None);
        let program = custom_program(payload.clone());
        let result = run(&mut engine, id, program);
        assert_eq!(result.error, None);
        assert_eq!(
            result.ruleset_publication.unwrap().names,
            Vec::<String>::new()
        );
        let Some(pb::command_output::Kind::Extraction(extract)) = &result.outputs[0].kind else {
            panic!("not an extract")
        };
        let output = extract.roots[0].variants[0].term;
        let node = &result.nodes[output as usize];
        assert!(
            matches!(&result.sorts[node.sort_id as usize].kind, Some(pb::sort::Kind::Family(f)) if f.name == "PyObject" && f.args.is_empty())
        );
        assert_eq!(
            node.kind,
            Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Custom(pb::CustomValue {
                    payload: payload.clone(),
                    args: vec![]
                }))
            }))
        );
        Python::attach(|py| {
            assert_eq!(
                load(py, &PyPickledValue(payload))
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "hello"
            )
        });
    }

    #[test]
    fn custom_dict_bytes_survive_clone_fresh_handle_and_destroy() {
        Python::initialize();
        let payload = Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("answer", 42).unwrap();
            dict.set_item("text", "owned").unwrap();
            dump(dict).unwrap().0
        });
        let mut engine = Engine::with_extension(Arc::new(PyObjectExtension::default()));
        let id = create(&mut engine);
        let result = run(&mut engine, id, custom_program(payload.clone()));
        let replay = extracted_program(&result, &payload);
        let clone = pb::CloneEGraphResponse::decode(
            engine
                .clone_egraph(&pb::CloneEGraphRequest { egraph_id: id }.encode_to_vec())
                .unwrap()
                .as_slice(),
        )
        .unwrap()
        .egraph_id;
        let fresh = create(&mut engine);
        for handle in [id, clone, fresh] {
            let result = run(&mut engine, handle, replay.clone());
            extracted_program(&result, &payload);
            engine
                .destroy(&pb::DestroyEGraphRequest { egraph_id: handle }.encode_to_vec())
                .unwrap();
        }
        drop(engine);
        let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
            value: Some(pb::primitive_value::Value::Custom(value)),
        })) = &replay.nodes[0].kind
        else {
            panic!("not Custom")
        };
        Python::attach(|py| {
            let loaded = load(py, &PyPickledValue(value.payload.clone())).unwrap();
            assert_eq!(
                loaded.get_item("answer").unwrap().extract::<i64>().unwrap(),
                42
            );
            assert_eq!(
                loaded
                    .get_item("text")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "owned"
            );
        });
    }

    #[test]
    fn custom_inspection_preparation_and_extraction_never_load_pickle() {
        Python::initialize();
        let payload = Python::attach(|py| {
            let module = PyModule::from_code(py, c"import builtins\nbuiltins._egglog_value_gate_loads = 0\ndef rebuild():\n    builtins._egglog_value_gate_loads += 1\n    return 'loaded explicitly'\nclass Probe:\n    def __reduce__(self):\n        return (rebuild, ())\n", c"value_gate.py", c"_egglog_value_gate").unwrap();
            dump(module.getattr("Probe").unwrap().call0().unwrap())
                .unwrap()
                .0
        });
        let value = PyPickledValue(payload.clone());
        assert!(format!("{value:?}").starts_with("PyPickledValue("));
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hash);
        let _ = hash.finish();
        assert_eq!(value, value.clone());
        let extension = Arc::new(PyObjectExtension::default());
        let mut engine = Engine::with_extension(extension.clone());
        let id = create(&mut engine);
        let original = custom_program(payload.clone());
        let mut lowered = original.clone();
        let mut graph = egglog_experimental::new_experimental_egraph();
        extension.initialize(&mut graph).unwrap();
        let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
            value: Some(pb::primitive_value::Value::Custom(custom)),
        })) = &original.nodes[0].kind
        else {
            unreachable!()
        };
        extension
            .lower_custom_value(
                &graph,
                graph.get_sort_by_name("PyObject").unwrap(),
                custom,
                None,
                &mut lowered,
            )
            .unwrap();
        assert_eq!(lowered.nodes[..original.nodes.len()], original.nodes);
        assert_eq!(lowered.sorts[..original.sorts.len()], original.sorts);
        assert_eq!(lowered.declarations, original.declarations);
        assert_eq!(lowered.nodes.len(), original.nodes.len() + 1);
        let mut unused = original.clone();
        unused.commands.clear();
        assert_eq!(run(&mut engine, id, unused).error, None);
        let mut invalid_later = original.clone();
        invalid_later.commands.push(pb::Command {
            kind: Some(pb::command::Kind::Extract(pb::Extract {
                roots: vec![99],
                variants: 1,
                extractor: pb::Extractor::Tree.into(),
                ..Default::default()
            })),
            ..Default::default()
        });
        let rejected = run(&mut engine, id, invalid_later);
        assert_eq!(
            rejected.error.unwrap().code,
            pb::ErrorCode::InvalidProgram as i32
        );
        assert!(rejected.outputs.is_empty());
        assert!(rejected.ruleset_publication.unwrap().names.is_empty());
        let result = run(&mut engine, id, original);
        let replay = extracted_program(&result, &payload);
        extracted_program(&run(&mut engine, id, replay), &payload);
        Python::attach(|py| {
            _export_pyobject_catalog(py).unwrap();
            let builtins = PyModule::import(py, "builtins").unwrap();
            assert_eq!(
                builtins
                    .getattr("_egglog_value_gate_loads")
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                0
            );
            assert_eq!(
                load(py, &value).unwrap().extract::<String>().unwrap(),
                "loaded explicitly"
            );
            assert_eq!(
                builtins
                    .getattr("_egglog_value_gate_loads")
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                1
            );
            builtins.delattr("_egglog_value_gate_loads").unwrap();
        });
        let bad_pickle = b"not a pickle".to_vec();
        assert!(format!("{:?}", PyPickledValue(bad_pickle.clone())).contains("PyPickledValue"));
        extracted_program(
            &run(&mut engine, id, custom_program(bad_pickle.clone())),
            &bad_pickle,
        );
        Python::attach(|py| assert!(load(py, &PyPickledValue(bad_pickle)).is_err()));
    }

    #[test]
    fn custom_rejection_is_before_authored_effects() {
        use egglog_experimental::protobuf::source::Source;
        let phases = Source::new(
            None,
            "(function observed () i64 :merge new) (set (observed) 0) (check (= (observed) 0))",
        )
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        for configured in [false, true] {
            let mut engine = if configured {
                Engine::with_extension(Arc::new(PyObjectExtension::default()))
            } else {
                Engine::default()
            };
            let id = create(&mut engine);
            for phase in &phases[..2] {
                assert_eq!(run(&mut engine, id, phase.clone()).error, None);
            }
            for invalid in 0..6 {
                let mut program = custom_program(b"inert invalid pickle".to_vec());
                program.sorts.push(pb::Sort {
                    kind: Some(pb::sort::Kind::Family(pb::HostSort {
                        name: "i64".into(),
                        args: vec![],
                    })),
                    ..Default::default()
                });
                program.nodes.push(pb::Node {
                    sort_id: 1,
                    kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                        value: Some(pb::primitive_value::Value::I64(1)),
                    })),
                    ..Default::default()
                });
                program.commands = vec![pb::Command {
                    kind: Some(pb::command::Kind::Action(pb::Action {
                        kind: Some(pb::action::Kind::Set(pb::Set {
                            target: Some(pb::Call {
                                func: "observed".into(),
                                args: vec![],
                            }),
                            value: Some(1),
                        })),
                        ..Default::default()
                    })),
                    ..Default::default()
                }];
                match invalid {
                    0 => {
                        let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                            value: Some(pb::primitive_value::Value::Custom(value)),
                        })) = &mut program.nodes[0].kind
                        else {
                            unreachable!()
                        };
                        value.args.push(1);
                    }
                    1 => program.nodes[0].sort_id = 1,
                    2 => {
                        let Some(pb::sort::Kind::Family(family)) = &mut program.sorts[0].kind
                        else {
                            unreachable!()
                        };
                        family.args.push(1);
                    }
                    3 => program.commands.push(pb::Command {
                        kind: Some(pb::command::Kind::Extract(pb::Extract {
                            roots: vec![99],
                            variants: 1,
                            extractor: pb::Extractor::Tree.into(),
                            ..Default::default()
                        })),
                        ..Default::default()
                    }),
                    4 => {
                        program.sorts[0].kind = Some(pb::sort::Kind::Eq("PyObject".into()));
                        program.declarations.push(pb::Declaration {
                            kind: Some(pb::declaration::Kind::EqSort(pb::EqSort {
                                name: "PyObject".into(),
                                ..Default::default()
                            })),
                            ..Default::default()
                        });
                    }
                    _ => program.declarations.push(pb::Declaration {
                        kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
                            name: "PyObject".into(),
                            arity: 1,
                            bindings: None,
                        })),
                        ..Default::default()
                    }),
                }
                let response = run(&mut engine, id, program);
                assert_eq!(
                    response.error.unwrap().code,
                    pb::ErrorCode::InvalidProgram as i32,
                    "configured={configured}, invalid={invalid}"
                );
                assert!(response.outputs.is_empty());
                assert!(response.ruleset_publication.unwrap().names.is_empty());
                assert_eq!(
                    run(&mut engine, id, phases[2].clone()).error,
                    None,
                    "no preceding Set may run"
                );
            }
        }
    }

    #[test]
    fn nested_vec_custom_bytes_roundtrip() {
        let payload = b"inert nested bytes".to_vec();
        let mut program = custom_program(payload.clone());
        program.sorts.push(pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "Vec".into(),
                args: vec![0],
            })),
            ..Default::default()
        });
        program.nodes.push(pb::Node {
            sort_id: 1,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Vec(pb::ValueList {
                    items: vec![0, 0],
                })),
            })),
            ..Default::default()
        });
        let Some(pb::command::Kind::Extract(extract)) = &mut program.commands[0].kind else {
            unreachable!()
        };
        extract.roots = vec![1];
        let mut engine = Engine::with_extension(Arc::new(PyObjectExtension::default()));
        let id = create(&mut engine);
        for _ in 0..2 {
            let result = run(&mut engine, id, program);
            assert_eq!(result.error, None);
            assert!(
                result
                    .ruleset_publication
                    .as_ref()
                    .unwrap()
                    .names
                    .is_empty()
            );
            let Some(pb::command_output::Kind::Extraction(extract)) = &result.outputs[0].kind
            else {
                unreachable!()
            };
            let root = extract.roots[0].variants[0].term;
            let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Vec(values)),
            })) = &result.nodes[root as usize].kind
            else {
                panic!("not a Vec")
            };
            assert_eq!(values.items.len(), 2);
            assert_eq!(values.items[0], values.items[1]);
            let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Custom(value)),
            })) = &result.nodes[values.items[0] as usize].kind
            else {
                panic!("not Custom")
            };
            assert_eq!(value.payload, payload);
            program = pb::Program {
                ir_version: 1,
                nodes: result.nodes.clone(),
                sorts: result.sorts.clone(),
                commands: vec![pb::Command {
                    kind: Some(pb::command::Kind::Extract(pb::Extract {
                        roots: vec![root],
                        variants: 1,
                        extractor: pb::Extractor::Tree.into(),
                        ..Default::default()
                    })),
                    ..Default::default()
                }],
                ..Default::default()
            };
        }
    }

    // Validates the whole observation boundary and returns an independently
    // owned, re-ingestible arena. This is not an out-of-band result channel.
    fn extracted_program(result: &pb::RunProgramResponse, payload: &[u8]) -> pb::Program {
        assert_eq!(result.error, None);
        assert!(
            result
                .ruleset_publication
                .as_ref()
                .unwrap()
                .names
                .is_empty()
        );
        let Some(pb::command_output::Kind::Extraction(extract)) = &result.outputs[0].kind else {
            panic!("not extraction")
        };
        let extracted = &extract.roots[0].variants[0];
        let node = &result.nodes[extracted.term as usize];
        assert!(
            matches!(&result.sorts[node.sort_id as usize].kind, Some(pb::sort::Kind::Family(family)) if family.name == "PyObject" && family.args.is_empty())
        );
        assert_eq!(
            node.kind,
            Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Custom(pb::CustomValue {
                    payload: payload.to_vec(),
                    args: vec![]
                }))
            }))
        );
        assert!(
            matches!(&result.nodes[extracted.cost.unwrap() as usize].kind, Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue { value: Some(pb::primitive_value::Value::I64(cost)) })) if *cost > 0)
        );
        pb::Program {
            ir_version: 1,
            nodes: result.nodes.clone(),
            sorts: result.sorts.clone(),
            commands: vec![pb::Command {
                kind: Some(pb::command::Kind::Extract(pb::Extract {
                    roots: vec![extracted.term],
                    variants: 1,
                    extractor: pb::Extractor::Tree.into(),
                    ..Default::default()
                })),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn create(engine: &mut Engine) -> u64 {
        let request = pb::CreateEGraphRequest {
            sorts: vec![pb::Sort {
                kind: Some(pb::sort::Kind::Family(pb::HostSort {
                    name: "i64".into(),
                    args: vec![],
                })),
                ..Default::default()
            }],
            options: Some(pb::EGraphOptions {
                cost_sort: Some(0),
                ..Default::default()
            }),
            ..Default::default()
        };
        pb::CreateEGraphResponse::decode(
            engine.create(&request.encode_to_vec()).unwrap().as_slice(),
        )
        .unwrap()
        .egraph_id
    }

    fn run(engine: &mut Engine, id: u64, program: pb::Program) -> pb::RunProgramResponse {
        let request = pb::RunProgramRequest {
            egraph_id: id,
            program: Some(program),
            profile: true,
        };
        pb::RunProgramResponse::decode(engine.run(&request.encode_to_vec()).unwrap().as_slice())
            .unwrap()
    }

    fn custom_program(payload: Vec<u8>) -> pb::Program {
        pb::Program {
            ir_version: 1,
            sorts: vec![pb::Sort {
                kind: Some(pb::sort::Kind::Family(pb::HostSort {
                    name: "PyObject".into(),
                    args: vec![],
                })),
                ..Default::default()
            }],
            nodes: vec![pb::Node {
                sort_id: 0,
                kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                    value: Some(pb::primitive_value::Value::Custom(pb::CustomValue {
                        payload,
                        args: vec![],
                    })),
                })),
                ..Default::default()
            }],
            commands: vec![pb::Command {
                kind: Some(pb::command::Kind::Extract(pb::Extract {
                    roots: vec![0],
                    variants: 1,
                    extractor: pb::Extractor::Tree.into(),
                    ..Default::default()
                })),
                ..Default::default()
            }],
            ..Default::default()
        }
    }
}

#[pymethods]
impl ProtoEngine {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn create<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::create)
    }

    fn run<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::run)
    }

    fn clone_graph<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::clone_egraph)
    }

    fn destroy<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::destroy)
    }

    fn configure_resources<'py>(
        &mut self,
        py: Python<'py>,
        request: &Bound<'py, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.request(py, request, Engine::configure_resources)
    }
}
