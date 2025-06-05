// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <functional>
#include <stack>
#include <string>
#include <vector>

namespace cosmos_reason1 {

struct WatchdogAction {
  std::string name_;
  int64_t comm_idx_;
  std::function<void(int64_t comm_idx)> abort_func_;
};

struct WatchdogContext {
  std::vector<WatchdogAction> actions_;
};

class WatchdogTLS {
 private:
  static inline thread_local std::stack<WatchdogContext*> context_stack_;

 public:
  static WatchdogContext* new_context() {
    context_stack_.push(new WatchdogContext());
    return context_stack_.top();
  }

  static void pop_context(bool abort) {
    if (context_stack_.empty()) {
      return;
    }

    auto context = context_stack_.top();
    context_stack_.pop();
    if (abort) {
      for (auto& action : context->actions_) {
        if (action.abort_func_) {
          action.abort_func_(action.comm_idx_);
        }
      }
    }
    delete context;
  }

  static void add_action(const WatchdogAction& action) {
    if (context_stack_.empty()) {
      return;
    }
    WatchdogContext* context = context_stack_.top();
    context->actions_.push_back(action);
  }
};

}  // namespace cosmos_reason1