#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include "problem/heuristic.h"

Heuristic::Heuristic(double runtime_limit, bool validation) :
  validation_(validation),
  best_(0.0),
  runtime_limit_(runtime_limit) {
  start_time_ = clock();
}

double Heuristic::Runtime() {
  clock_t tv = clock();
  double secs = (double)(tv - start_time_) / CLOCKS_PER_SEC;
  return secs;
}

std::string Heuristic::History() {
  std::stringstream out_str;
  out_str << std::setprecision(15) << "[";
  for (int i = 0; i < past_solution_values_.size(); i++) {
    if (i > 0) out_str << ";";
    out_str << past_solution_values_[i] << ":" << past_solution_times_[i];
  }
  out_str << "]";
  return out_str.str();
}
