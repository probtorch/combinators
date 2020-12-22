-- | quick test of JW's semantics

module Main where

type Tensor = ()
type Output = ()

type Weight = Tensor
type ProgramType a b = a -> b
type KernelType a b = (ProgramType a b) -> b

data Target a b
  = Program (ProgramType a b )
  | Kernel (KernelType a b )
  | Reverse (Target a b) (KernelType a b)

data Proposal a b
  = Propose (Target a b) (Proposal a b)
  | Resample (Proposal a b)
  | Forward (KernelType a b) (Proposal a b)
  | AsTarget (Target a b)
