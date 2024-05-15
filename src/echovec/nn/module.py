"""A tiny ``Module`` system in the spirit of the PyTorch one.

Parameters are :class:`Tensor` objects with ``requires_grad=True``.  Modules
discover their parameters and sub-modules by inspecting ``__dict__``, plus an
explicit :class:`ModuleList` for ordered collections.  No metaclass magic.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from ..autograd import Tensor


def Parameter(data, requires_grad: bool = True) -> Tensor:
    """Create a trainable tensor (a thin, explicit constructor)."""
    return Tensor(data, requires_grad=requires_grad)


class Module:
    """Base class for all neural-network building blocks."""

    def __init__(self) -> None:
        self._training = True

    # -- forward plumbing -----------------------------------------------------
    def forward(self, *args, **kwargs):  # pragma: no cover - overridden
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # -- parameter / submodule discovery -------------------------------------
    def _members(self) -> Iterator[tuple[str, object]]:
        for name, value in vars(self).items():
            if name.startswith("__"):
                continue
            yield name, value

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        for name, value in self._members():
            path = f"{prefix}{name}"
            if isinstance(value, Tensor):
                if value.requires_grad:
                    yield path, value
            elif isinstance(value, Module):
                # Module covers ModuleList too; both override as needed.
                yield from value.named_parameters(prefix=f"{path}.")

    def parameters(self) -> Iterator[Tensor]:
        for _, param in self.named_parameters():
            yield param

    def num_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters()))

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    # -- train / eval state ---------------------------------------------------
    @property
    def training(self) -> bool:
        return self._training

    def train(self, mode: bool = True) -> Module:
        self._training = mode
        for _, value in self._members():
            if isinstance(value, Module):
                value.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    # -- checkpointing --------------------------------------------------------
    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: param.data.copy() for name, param in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        params = dict(self.named_parameters())
        missing = set(params) - set(state)
        if missing:
            raise KeyError(f"missing parameters in state dict: {sorted(missing)}")
        for name, param in params.items():
            value = np.asarray(state[name])
            if value.shape != param.shape:
                raise ValueError(f"shape mismatch for '{name}': {value.shape} != {param.shape}")
            param.data = value.astype(param.data.dtype)


class ModuleList(Module):
    """An ordered list of sub-modules whose parameters are all discoverable."""

    def __init__(self, modules: Iterable[Module] | None = None) -> None:
        super().__init__()
        self._modules: list[Module] = list(modules or [])

    def append(self, module: Module) -> ModuleList:
        self._modules.append(module)
        return self

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int) -> Module:
        return self._modules[index]

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        for i, module in enumerate(self._modules):
            yield from module.named_parameters(prefix=f"{prefix}{i}.")

    def train(self, mode: bool = True) -> ModuleList:
        self._training = mode
        for module in self._modules:
            module.train(mode)
        return self
