# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Client-side ES2022 roofline and on-chip VMEM physics evaluator."""

from __future__ import annotations

from .assets import load_asset


def render_workbench_engine_js() -> str:
    """Return dependency-free ES2022 evaluator script hydrated from #advisor-data."""
    return load_asset("workbench_engine.js")
