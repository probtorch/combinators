-- | quick dsl prototype of what is going on in combinators
module Main where

import Data.Map
import Control.Monad.State.Strict

main :: IO ()
main = undefined

newtype Trace a = Trace (Map String a)

instance Semigroup (Trace a) where
  (Trace l) <> (Trace r) = Trace ( l <> r )

instance Monoid (Trace a) where
  mempty = Trace mempty
  mappend = (<>)

---------------------------------------------------
-- | work in progress
data TraceTree a
  = Leaf [(String, a)]
  | Branch [(String, a)] [TraceTree a]

instance Semigroup (TraceTree a) where

instance Monoid (TraceTree a) where
  mempty = Leaf []
  mappend = (<>)




-----------------------------------------------------------
newtype TraceM a x = TraceM (State (Trace (Map String a)) x)
