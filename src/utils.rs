use pyo3::prelude::*;

// Create a dataclass-like repr, of the name of the class of the object
// called with the repr of the fields
pub fn data_repr<T: pyo3::PyClass>(
    py: Python,
    slf: PyRef<T>,
    field_names: Vec<&str>,
) -> PyResult<String> {
    let binding = slf.into_pyobject(py)?;
    let obj = binding.as_any();
    let class_name: String = obj.getattr("__class__")?.getattr("__name__")?.extract()?;
    let field_strings: PyResult<Vec<String>> = field_names
        .iter()
        .map(|name| {
            obj.getattr(*name)
                .and_then(|x| x.call_method("__repr__", (), None)?.extract())
        })
        .collect();
    Ok(format!("{}({})", class_name, field_strings?.join(", ")))
}

// Macro to create a wrapper around rust enums.
// We create Python classes for each variant of the enum
// and create a wrapper enum around all variants to enable conversion to/from Python
// and to/from egglog
#[macro_export]
macro_rules! convert_enums {
    ($(
        $from_type:ty: $str:literal $($trait_outer:ty)? => $to_type:ident {
            $(
                $variant:ident$([name=$py_name:literal])?$([trait=$trait_inner:ty])?($($field:ident: $field_type:ty),*)
                $from_ident:ident -> $from:expr,
                $to_pat:pat => $to:expr
            );*
        }
    );*) => {
        $($(
            #[pyclass(frozen$(, name=$py_name)?)]
            #[derive(Clone, PartialEq, Eq$(, $trait_inner)?)]
            pub struct $variant {
                $(
                    #[pyo3(get)]
                    $field: $field_type,
                )*
            }

            #[pymethods]
            impl $variant {
                #[new]
                #[pyo3(signature=($($field),*))]
                fn new($($field: $field_type),*) -> Self {
                    Self {
                        $($field),*
                    }
                }

                fn __repr__(slf: PyRef<'_, Self>, py: Python) -> PyResult<String> {
                    data_repr(py, slf, vec![$(stringify!($field)),*])
                }

                fn __str__(&self) -> String {
                    format!($str, <$from_type>::from(self.clone()))
                }
                fn __richcmp__(&self, other: &Self, op: pyo3::basic::CompareOp, py: Python<'_>) -> PyResult<PyObject> {
                    match op {
                        pyo3::basic::CompareOp::Eq => Ok((self == other).into_pyobject(py)?.as_any().clone().unbind()),
                        pyo3::basic::CompareOp::Ne => Ok((self != other).into_pyobject(py)?.as_any().clone().unbind()),
                        _ => Ok(py.NotImplemented()),
                    }
                }
            }

            impl From<$variant> for $from_type {
                fn from($from_ident: $variant) -> $from_type {
                    $from
                }
            }
            impl From<&$variant> for $from_type {
                fn from($from_ident: &$variant) -> $from_type {
                    $from
                }
            }
        )*

        #[derive(FromPyObject, Clone, PartialEq, Eq$(, $trait_outer)?)]
        pub enum $to_type {
            $(
                $variant($variant),
            )*
        }
        impl<'py> IntoPyObject<'py> for $to_type {
            type Target = PyAny; // the Python type
            type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
            type Error = pyo3::PyErr;

            fn into_pyobject(self, py: Python<'py>) -> Result<Bound<'py, Self::Target>, Self::Error> {
                Ok(match self {
                    $(
                        $to_type::$variant(v) => v.into_pyobject(py)?.as_any().clone(),
                    )*
                })
            }
        }
        impl From<$to_type> for $from_type {
            fn from(other: $to_type) -> Self {
                match other {
                    $(
                        $to_type::$variant(v) => v.into(),
                    )*
                }
            }
        }

        impl From<$from_type> for $to_type {
            fn from(other: $from_type) -> Self {
                match &other {
                    $(
                        $to_pat => $to_type::$variant($to),
                    )*
                }
            }
        }

        impl From<&$to_type> for $from_type {
            fn from(other: &$to_type) -> Self {
                match other {
                    $(
                        $to_type::$variant(v) => v.into(),
                    )*
                }
            }
        }

        impl From<&$from_type> for $to_type {
            fn from(other: &$from_type) -> Self {
                match other {
                    $(
                        $to_pat => $to_type::$variant($to),
                    )*
                }
            }
        }

        impl From<&Box<$to_type>> for $from_type {
            fn from(other: &Box<$to_type>) -> Self {
                match &**other {
                    $(
                        $to_type::$variant(v) => v.into(),
                    )*
                }
            }
        }
        impl From<&Box<$from_type>> for $to_type {
            fn from(other: &Box<$from_type>) -> Self {
                match &**other {
                    $(
                        $to_pat => $to_type::$variant($to),
                    )*
                }
            }
        }
    )*
    pub fn add_enums_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
        $(
            $(
                module.add_class::<$variant>()?;
            )*
        )*
        Ok(())
    }
    };
}

#[macro_export]
macro_rules! convert_struct {
    ($(
        $from_type:ty: $str:literal $($struct_trait:ty)? => $to_type:ident($($field:ident: $field_type:ty$( = $default:expr)?),*)
            $from_ident:ident -> $from:expr,
            $to_ident:ident -> $to:expr
    );*) => {
        $(
            #[pyclass(frozen)]
            #[derive(Clone, PartialEq, Eq$(, $struct_trait)?)]
            pub struct $to_type {
                $(
                    #[pyo3(get)]
                    $field: $field_type,
                )*
            }

            #[pymethods]
            impl $to_type {
                #[new]
                #[pyo3(signature=($($field $(= $default)?),*))]
                fn new($($field: $field_type),*) -> Self {
                    Self {
                        $($field),*
                    }
                }

                fn __repr__(slf: PyRef<'_, Self>, py: Python) -> PyResult<String> {
                    data_repr(py, slf, vec![$(stringify!($field)),*])
                }
                fn __str__(&self) -> String {
                    format!($str, <$from_type>::from(self.clone()))
                }
                fn __richcmp__(&self, other: &Self, op: pyo3::basic::CompareOp, py: Python<'_>) -> PyResult<PyObject> {
                    Ok(match op {
                        pyo3::basic::CompareOp::Eq => (self == other).into_pyobject(py)?.as_any().clone().unbind(),
                        pyo3::basic::CompareOp::Ne => (self != other).into_pyobject(py)?.as_any().clone().unbind(),
                        _ => py.NotImplemented(),
                    })
                }
            }

            impl From<&$to_type> for $from_type {
                fn from($from_ident: &$to_type) -> $from_type {
                    $from
                }
            }
            impl From<&$from_type> for $to_type {
                fn from($to_ident: &$from_type) -> Self {
                    $to
                }
            }
            impl From<$to_type> for $from_type {
                fn from($from_ident: $to_type) -> $from_type {
                    $from
                }
            }
            impl From<$from_type> for $to_type {
                fn from($to_ident: $from_type) -> Self {
                    $to
                }
            }
        )*
        pub fn add_structs_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
            $(
                module.add_class::<$to_type>()?;
            )*
            Ok(())
        }
    };
}
pub use convert_enums;
pub use convert_struct;
