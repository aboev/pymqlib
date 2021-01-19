#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <string>
#include "heuristics/heuristic_factory.h"
#include "heuristics/maxcut/hyperheuristic.h"
#include "metrics/max_cut_metrics.h"
#include "problem/heuristic.h"
#include "problem/max_cut_instance.h"
#include "problem/qubo_instance.h"

// Instance wrapper -- store an instance (Max-Cut or QUBO). If it needed to
// be converted to the other to run on some heuristic, then store that converted
// version as well.
typedef struct {
  PyObject_HEAD
  MaxCutInstance *mi;
  QUBOInstance *qi;
  char itype; // 'M' or 'Q'
} Inst;

// Create an instance with all null values
static PyObject* Inst_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  Inst* self;
  self = (Inst*)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->mi = NULL;
    self->qi = NULL;
    self->itype = '\0';
  }
  return (PyObject*)self;
}

// Load an instance from a file
static int Inst_init(Inst* self, PyObject* args) {
  const char *itype;
  PyObject* left;
  PyObject* right;
  PyObject* data;
  PyArrayObject* aleft;
  PyArrayObject* aright;
  PyArrayObject* adata;
  int n;  // Number of nodes / variables
  if (!PyArg_ParseTuple(args, "sOOOi", &itype, &left, &right, &data, &n)) {
    return -1;
  }
  if (strcmp(itype, "M") && strcmp(itype, "Q")) {
    PyErr_Format(PyExc_ValueError, "invalid instance type %s", itype);
    return -1;
  }
  self->itype = itype[0];

  // Load the tuple representation of sparse matrix -- left < right are the
  // arrays that indicate each from/to pairing and data contains their weight.
  if (!PyArray_Check(left) || !PyArray_Check(right) || !PyArray_Check(data)) {
    PyErr_Format(PyExc_TypeError, "unexpected object type -- want numpy array");
    return -1;
  }
  if (n <= 0) {
    PyErr_Format(PyExc_ValueError, "invalid instance size %d", n);
    return -1;
  }
  aleft = (PyArrayObject*)left;
  aright = (PyArrayObject*)right;
  adata = (PyArrayObject*)data;
  if (PyArray_SIZE(aleft) != PyArray_SIZE(aright) ||
      PyArray_SIZE(aleft) != PyArray_SIZE(adata)) {
    PyErr_Format(PyExc_ValueError, "sparse rep arrays should be of same length");
    return -1;
  }

  // Loop in lockstep through the three arrays to build an edgeList
  // object, which is the expected data input format for building instances.
  PyArray_Descr *dtype_i = PyArray_DescrFromType(NPY_INT32);
  PyArray_Descr *dtype_d = PyArray_DescrFromType(NPY_DOUBLE);
  NpyIter *iter_l = NpyIter_New(aleft, NPY_ITER_READONLY, NPY_KEEPORDER,
				NPY_NO_CASTING, dtype_i);
  NpyIter *iter_r = NpyIter_New(aright, NPY_ITER_READONLY, NPY_KEEPORDER,
				NPY_NO_CASTING, dtype_i);
  NpyIter *iter_d = NpyIter_New(adata, NPY_ITER_READONLY, NPY_KEEPORDER,
				NPY_NO_CASTING, dtype_d);
  NpyIter_IterNextFunc *iternext_l = NpyIter_GetIterNext(iter_l, NULL);
  NpyIter_IterNextFunc *iternext_r = NpyIter_GetIterNext(iter_r, NULL);
  NpyIter_IterNextFunc *iternext_d = NpyIter_GetIterNext(iter_d, NULL);
  int** dataptr_l = (int**)NpyIter_GetDataPtrArray(iter_l);
  int** dataptr_r = (int**)NpyIter_GetDataPtrArray(iter_r);
  double** dataptr_d = (double**)NpyIter_GetDataPtrArray(iter_d);
  std::vector<Instance::InstanceTuple> edgeList; // Sparse representation
  std::vector<double> mainDiag(n, 0.0);  // Main diagonal (QUBO instances only)
  do {
    if (**dataptr_l < 1 || **dataptr_l > n || **dataptr_r < 1 || **dataptr_r > n) {
      PyErr_Format(PyExc_ValueError,
		   "Expect 1-indexed values; got link %d-%d in instance of size %d",
		   **dataptr_l, **dataptr_r, n);
      NpyIter_Deallocate(iter_l);
      NpyIter_Deallocate(iter_r);
      NpyIter_Deallocate(iter_d);
      return -1;
    } else if (self->itype == 'Q' && **dataptr_l == **dataptr_r) {
      mainDiag[**dataptr_l-1] = **dataptr_d;
    } else if (**dataptr_l != **dataptr_r) {
      edgeList.push_back(Instance::InstanceTuple(std::pair<int, int>(**dataptr_l, **dataptr_r), **dataptr_d));  // 1-indexed sparse rep
    }
  } while (iternext_l(iter_l) && iternext_r(iter_r) && iternext_d(iter_d));
  NpyIter_Deallocate(iter_l);
  NpyIter_Deallocate(iter_r);
  NpyIter_Deallocate(iter_d);

  // Build the instance
  if (self->itype == 'M') {
    self->mi = new MaxCutInstance(edgeList, n);
  } else {
    self->qi = new QUBOInstance(edgeList, mainDiag, n);
  }
  return 0;
}

// Clean up allocated memory for our instance
static void Inst_dealloc(Inst* self) {
  if (self->mi != NULL) {
    delete self->mi;
    self->mi = NULL;
  }
  if (self->qi != NULL) {
    delete self->qi;
    self->qi = NULL;
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// We opt for this verbose version of the PyTypeObject definition to
// make the Visual C++ compiler happy; got the template from
// https://docs.python.org/3/c-api/typeobj.html
static PyTypeObject InstType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_MQLib._Inst",                 /* tp_name */
    sizeof(Inst),                   /* tp_basicsize */
    0,                              /* tp_itemsize */
    (destructor)Inst_dealloc,       /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    0,                              /* tp_repr */
    0,                              /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    0,                              /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    "Instance wrapper",             /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    0,                              /* tp_methods */
    0,                              /* tp_members */
    0,                              /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)Inst_init,            /* tp_init */
    0,                              /* tp_alloc */
    Inst_new,                       /* tp_new */
};

// See https://stackoverflow.com/a/52732077/3093387 -- free our allocated
// memory once the parent numpy array is deleted
void capsule_cleanup(PyObject *capsule) {
  void *memory = PyCapsule_GetPointer(capsule, NULL);
  free(memory);
}

bool pyCallback(PyObject *callback, const std::vector<int>& assignments) {
  PyObject *list = PyList_New(assignments.size());
  for (size_t i = 0; i < assignments.size(); ++i) {
    PyList_SET_ITEM(list, i, PyLong_FromLong(assignments[i]));
  }
  PyObject *arglist = Py_BuildValue("(O)", list);
  PyObject* res = PyObject_CallObject(callback, arglist);
  long result = PyLong_AsLong(res);
  if (res != NULL) {
    Py_DECREF(res);
  }
  Py_DECREF(arglist);
  Py_DECREF(list);
  if (result != 0) result = 1;
  return result;
}

class PyQUBOCallback : public QUBOCallback {
  PyObject *callback;
  double timeout;
  public:
    PyQUBOCallback (PyObject *callback, double timeout): callback{ callback }, timeout{ timeout } {}
    bool Report(const QUBOSimpleSolution& solution, bool newBest, 
          double runtime) {
      bool res = 1;
      if (newBest == 1) res = pyCallback(callback, solution.get_assignments());
      if ((runtime >= timeout) && (timeout > 0)) res = 0;
      return res;
    }
    bool Report(const QUBOSimpleSolution& solution, bool newBest,
            double runtime, int iter) {
      bool res = 1;
      if (newBest == 1) res = pyCallback(callback, solution.get_assignments());
      if ((runtime >= timeout) && (timeout > 0)) res = 0;
      return res;
    }
};

class PyMaxCutCallback : public MaxCutCallback {
  PyObject *callback;
  double timeout;
  public:
    PyMaxCutCallback (PyObject *callback, double timeout): callback{ callback }, timeout{ timeout } {}
    bool Report(const MaxCutSimpleSolution& solution, bool newBest,
	  double runtime) {
      bool res = 1;
      if (newBest == 1) res = pyCallback(callback, solution.get_assignments());
      if ((runtime >= timeout) && (timeout > 0)) res = 0;
      return res;
    }
    bool Report(const MaxCutSimpleSolution& solution, bool newBest,
	  double runtime, int iter) {
      bool res  = 1;
      if (newBest == 1) res = pyCallback(callback, solution.get_assignments());
      if ((runtime >= timeout) && (timeout > 0)) res = 0;
      return res;
    }
};

// Run a heuristic (Many parts of this closely match to sections of main.cpp)
static PyObject* runHeuristic(PyObject *self, PyObject *args) {
  // Load arguments
  PyObject *_inst;
  PyObject *callback;
  PyObject *py_callback = NULL;
  Inst *inst;
  const char *hname;  // Heuristic name
  MaxCutHeuristic *mh = NULL;
  QUBOHeuristic *qh = NULL;
  Heuristic *heuristic = NULL;
  double runtime;
  int seed;
  HeuristicFactory factory;
  std::string selected;  // If hyperheuristic is run, what heuristic was selected?
  int n;  // Instance size
  
  if (!PyArg_ParseTuple(args, "sOdOi", &hname, &_inst, &runtime, &callback, &seed)) {
    return NULL;
  }
  if (!PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "parameter must be callable");
    return NULL;
  }
  Py_XINCREF(callback);         /* Add a reference to new callback */
  Py_XDECREF(py_callback);  /* Dispose of previous callback */
  py_callback = callback;       /* Remember new callback */
  /* Boilerplate to return "None" */
  Py_INCREF(Py_None);

  selected = hname;
  if (!PyObject_IsInstance(_inst, (PyObject*)&InstType)) {
    PyErr_Format(PyExc_TypeError, "_Inst expected; got %s", _inst->ob_type->tp_name);
    return NULL;
  }
  inst = (Inst*)_inst;
  if (inst->itype == 'M' && inst->mi != NULL) {
    n = inst->mi->get_size();
  } else if (inst->itype == 'Q' && inst->qi != NULL) {
    n = inst->qi->get_size();
  } else {
    PyErr_Format(PyExc_ValueError, "Malformed _Inst passed");
    return NULL;
  }

  if (seed < 0) {
    seed = time(0);  // If seed not provided, use the current time
  }
  srand(seed);
  if (runtime < 0) {
    PyErr_Format(PyExc_ValueError, "invalid runtime limit");
    return NULL;
  }

  // Do the run itself
  if (factory.ValidMaxCutHeuristicCode(hname)) {
    if (!inst->mi) {
      inst->mi = new MaxCutInstance(*(inst->qi));
    }
    PyMaxCutCallback *cb = new PyMaxCutCallback(py_callback, runtime);
    mh = factory.RunMaxCutHeuristic(hname, *(inst->mi), runtime, false, cb);
    heuristic = mh;
  } else if (factory.ValidQUBOHeuristicCode(hname)) {
    if (!inst->qi) {
      inst->qi = new QUBOInstance(*(inst->mi));
    }
    PyQUBOCallback *cb = new PyQUBOCallback(py_callback, runtime);
    qh = factory.RunQUBOHeuristic(hname, *(inst->qi), runtime, false, cb);
    heuristic = qh;
  } else {
    PyErr_Format(PyExc_ValueError, "Illegal heuristic code %s", hname);
    return NULL;
  }
  if (!heuristic) {
    PyErr_Format(PyExc_RuntimeError, "Error running heuristic");
    return NULL;
  }

  // Construct our return values
  double bestObj = heuristic->get_best();
  PyObject *sln = NULL;
  PyObject *solVals = NULL;
  PyObject *solTimes = NULL;
  npy_intp dims[1];
  int* retsol = (int*)calloc(n+1, sizeof(int));  // Buffer to return solution
  double* retsv = (double*)calloc(heuristic->past_solution_values_.size(),
				  sizeof(double));
  double* retst = (double*)calloc(heuristic->past_solution_times_.size(),
				  sizeof(double));
  if (!retsol || !retsv || !retst) {
    PyErr_Format(PyExc_MemoryError, "Can't allocate memory to return solution");
    if (heuristic) {  delete heuristic; }
    if (retsol) {  free(retsol); }
    if (retsv) {  free(retsv); }
    if (retst) {  free(retst); }
    return NULL;
  }
  if (mh && inst->itype == 'M') {
    // Ran a max-cut heuristic on a max-cut instance; just return the assignments
    const MaxCutSimpleSolution& sol = mh->get_best_solution();
    dims[0] = sol.get_assignments().size();
    memcpy(retsol, sol.get_assignments().data(), dims[0]*sizeof(int));
  } else if (mh) {
    // Ran a max-cut heuristic on a QUBO instance; convert back the assignments
    QUBOSimpleSolution sol(mh->get_best_solution(), *(inst->qi), NULL);
    dims[0] = sol.get_assignments().size();
    memcpy(retsol, sol.get_assignments().data(), dims[0]*sizeof(int));
  } else if (qh && inst->itype == 'M') {
    // Ran a QUBO heuristic on a max-cut instance; convert back the assignments
    MaxCutSimpleSolution sol(qh->get_best_solution(), *(inst->mi), NULL);
    dims[0] = sol.get_assignments().size();
    memcpy(retsol, sol.get_assignments().data(), dims[0]*sizeof(int));
  } else {
    // Ran a QUBO heuristic on a QUBO instance; just return the assignments
    const QUBOSimpleSolution& sol = qh->get_best_solution();
    dims[0] = sol.get_assignments().size();
    memcpy(retsol, sol.get_assignments().data(), dims[0]*sizeof(int));
  }

  // Create numpy array for the solution and transfer ownership of the
  // raw data pointer (from the calloc above) to the numpy object
  // (see https://stackoverflow.com/a/52732077/3093387) so it is freed
  // once the python object is deleted
  sln = PyArray_SimpleNewFromData(1, dims, NPY_INT, retsol);
  PyObject* slnCapsule = PyCapsule_New(retsol, NULL, capsule_cleanup);
  PyArray_SetBaseObject((PyArrayObject*)sln, slnCapsule);

  // Repeat for the past solution values and times
  dims[0] = heuristic->past_solution_values_.size();
  memcpy(retsv, heuristic->past_solution_values_.data(),
	 dims[0]*sizeof(double));
  solVals = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, retsv);
  PyObject* svCapsule = PyCapsule_New(retsv, NULL, capsule_cleanup);
  PyArray_SetBaseObject((PyArrayObject*)solVals, svCapsule);

  dims[0] = heuristic->past_solution_times_.size();
  memcpy(retst, heuristic->past_solution_times_.data(),
	 dims[0]*sizeof(double));
  solTimes = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, retst);
  PyObject* stCapsule = PyCapsule_New(retst, NULL, capsule_cleanup);
  PyArray_SetBaseObject((PyArrayObject*)solTimes, stCapsule);
  
  // Free memory
  if (heuristic) {  delete heuristic; }

  // Return a tuple of values -- we use the "N" option instead of the
  // "O" option when passing our numpy arrays because we want to be returning
  // objects with refcount of 1, not 2 (so they are garbage collected once
  // they are deleted on the python side). They already have a refcount of 1
  // from object construction, so we don't want to increment it to 2 with tuple
  // construction ("O" would do this).
  return Py_BuildValue("sfNNN", selected.c_str(), bestObj, sln, solVals, solTimes);
}

// Get instance metrics
static PyObject* instanceMetrics(PyObject *self, PyObject *args) {
  PyObject* _inst;
  Inst *inst;
  if (!PyArg_ParseTuple(args, "O", &_inst)) {
    return NULL;
  }
  if (!PyObject_IsInstance(_inst, (PyObject*)&InstType)) {
    PyErr_Format(PyExc_TypeError, "_Inst expected; got %s", _inst->ob_type->tp_name);
    return NULL;
  }
  inst = (Inst*)_inst;

  // Get the values to output
  std::vector<std::string> metric_names;
  GraphMetrics::AllMetricNames(&metric_names);
  std::vector<std::string> runtime_names;
  GraphMetrics::AllRuntimeTypes(&runtime_names);

  // If this is a QUBO instance, convert it and then grab the metrics
  if (!inst->mi) {
    inst->mi = new MaxCutInstance(*(inst->qi));
  }
  std::vector<double> metrics;
  std::vector<double> runtimes;
  GraphMetrics gm(*(inst->mi));
  gm.AllMetrics(&metrics, &runtimes);

  if (metric_names.size() != metrics.size() || runtime_names.size() != runtimes.size()) {
    PyErr_Format(PyExc_RuntimeError, "Unexpected instance metric count");
    return NULL;
  }
  
  // Store in python lists (https://docs.python.org/2.0/api/refcountDetails.html)
  PyObject* MNlist = PyList_New(metric_names.size());
  for (int i=0; i < (int)metric_names.size(); ++i) {
    PyList_SetItem(MNlist, i, Py_BuildValue("s", metric_names[i].c_str()));
  }
  PyObject* Mlist = PyList_New(metrics.size());
  for (int i=0; i < (int)metrics.size(); ++i) {
    PyList_SetItem(Mlist, i, PyFloat_FromDouble(metrics[i]));
  }
  PyObject* RNlist = PyList_New(runtime_names.size());
  for (int i=0; i < (int)runtime_names.size(); ++i) {
    PyList_SetItem(RNlist, i, Py_BuildValue("s", runtime_names[i].c_str()));
  }
  PyObject* Rlist = PyList_New(runtimes.size());
  for (int i=0; i < (int)runtimes.size(); ++i) {
    PyList_SetItem(Rlist, i, PyFloat_FromDouble(runtimes[i]));
  }

  // Return -- we use N to not add an extra refcount
  return Py_BuildValue("NNNN", MNlist, Mlist, RNlist, Rlist);
}

// Return the set of all valid heuristics
static PyObject* getHeuristics(PyObject *self, PyObject *args) {
  // Load the names and descriptions of each Max-Cut and QUBO heuristic
  HeuristicFactory factory;
  const std::map<std::string, HeuristicFactory::MaxCutCreator>& max_cut_map =
    factory.max_cut_map_;
  const std::map<std::string, HeuristicFactory::QUBOCreator>& qubo_map =
    factory.qubo_map_;
  std::vector<std::string> max_cut_names;
  std::vector<std::string> max_cut_descs;
  for (auto iter = max_cut_map.begin(); iter != max_cut_map.end(); ++iter) {
    max_cut_names.push_back(iter->first);
  }
  std::sort(max_cut_names.begin(), max_cut_names.end());
  for (int i=0; i < (int)max_cut_names.size(); ++i) {
    max_cut_descs.push_back(max_cut_map.at(max_cut_names[i]).second);
  }
  std::vector<std::string> qubo_names;
  std::vector<std::string> qubo_descs;
  for (auto iter = qubo_map.begin(); iter != qubo_map.end(); ++iter) {
    qubo_names.push_back(iter->first);
  }
  std::sort(qubo_names.begin(), qubo_names.end());
  for (int i=0; i < (int)qubo_names.size(); ++i) {
    qubo_descs.push_back(qubo_map.at(qubo_names[i]).second);
  }

  // Create four python lists
  PyObject* MNlist = PyList_New(max_cut_names.size());
  for (int i=0; i < (int)max_cut_names.size(); ++i) {
    PyList_SetItem(MNlist, i, Py_BuildValue("s", max_cut_names[i].c_str()));
  }
  PyObject* MDlist = PyList_New(max_cut_descs.size());
  for (int i=0; i < (int)max_cut_descs.size(); ++i) {
    PyList_SetItem(MDlist, i, Py_BuildValue("s", max_cut_descs[i].c_str()));
  }
  PyObject* QNlist = PyList_New(qubo_names.size());
  for (int i=0; i < (int)qubo_names.size(); ++i) {
    PyList_SetItem(QNlist, i, Py_BuildValue("s", qubo_names[i].c_str()));
  }
  PyObject* QDlist = PyList_New(qubo_descs.size());
  for (int i=0; i < (int)qubo_descs.size(); ++i) {
    PyList_SetItem(QDlist, i, Py_BuildValue("s", qubo_descs[i].c_str()));
  }  

  // Return -- we use N to not add an extra refcount
  return Py_BuildValue("NNNN", MNlist, MDlist, QNlist, QDlist);
}

PyMethodDef method_table[] =
  {{"runHeuristic", (PyCFunction)runHeuristic, METH_VARARGS,
    "Given a heuristic and instance, do the run"},
   {"instanceMetrics", (PyCFunction)instanceMetrics, METH_VARARGS,
    "Get Max-Cut metrics for an instance"},
   {"getHeuristics", (PyCFunction)getHeuristics, METH_VARARGS,
    "Get all heuristic names and descriptions"},
   {NULL, NULL, 0, NULL}
};

PyModuleDef _MQLib_module = {
			     PyModuleDef_HEAD_INIT,
			     "_MQLib",
			     "MQLib C++ interface",
			     -1,
			     method_table
};


PyMODINIT_FUNC PyInit__MQLib(void) {
  PyObject *m;
  if (PyType_Ready(&InstType) < 0)
    return NULL;

  m = PyModule_Create(&_MQLib_module);
  if (m == NULL)
    return NULL;

  Py_INCREF(&InstType);
  if (PyModule_AddObject(m, "_Inst", (PyObject*)&InstType) < 0) {
    Py_DECREF(&InstType);
    Py_DECREF(m);
    return NULL;
  }

  // Allow numpy arrays
  _import_array();
  
  return m;
}
