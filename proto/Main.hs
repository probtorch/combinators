-- | This PPL contains two statements:
--
--     sample :: Address -> Dist -> PPL [Double]
--
-- Sample takes an address (which is a string), a distribution object and
-- returns a float.
--
--     observe :: Address -> Dist -> [Double] -> PPL ()
--
-- Observe takes an address, dist, and observation and alters state.
--
-- We rely on haskell primitives (like "return" and numeric operations) for most
-- other things.
--
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE NoImplicitPrelude #-}
module Main where

import Prelude hiding (putStrLn)
import Control.Monad.State
import qualified System.Random.MWC.Probability as MWC

import Lib

-- Assuming we have some kind of black box that can produce floats in a pure-way
type BlackBox = Double -> Double -> Double

-- we can write a program, like on page 226, as follows:
programOnPage226 :: [Double] -> BlackBox -> BlackBox -> Double -> PPL [Double]
programOnPage226 y ηy ηv θ = do
  z <- head <$> sample "z" (Multinomial 1 [θ])         -- we only care about the first sample here
  v <- head <$> sample "v" (Normal (ηy z θ) (ηv z θ))  -- ...and here
  observe "y" (Normal (ηy v θ) 1) y
  return [z, v]

type Prior = ()
type Out = ()

data InferenceF s a
  = Target Prior
  -- ^ A target can always be normalized and always has a prior distribution
  | Kernel (Out -> Out)
  -- ^ A kernel extends a target program. It can always evaluate the density (of
  --   the program?)  in a fully-deterministic way (if there are missing values,
  --   you sample from the program's prior).
  --
  --   they do _not_ compute likelihood weights or any other side effects --
  --   they will only check if a variable is in the trace. If it is, it will
  --   compute a log-prob and check to make sure there aren't any extra things
  --   in the trace. If not, then it will return 0 probability for all values.
  --
  --   important to note that this is not conditioned evaluation. It will use
  --   what it can and throw awaay the rest.
  --
  --   kernels sample from the prior, but when you eval there are slightly
  --   different rules.
  | GetMemberClubs String (Maybe [String] -> a)
  | GetInput (String -> a)
  | Display String a -- util function
  deriving (Functor)

main :: IO ()
main = do
  s <- MWC.createSystemSeed
  printPPL $ program
  print $ evalState (stateHandler program) (s, mempty)
  where
    program = programOnPage226 [1] (+) (-) 2

