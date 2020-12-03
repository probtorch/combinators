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


main :: IO ()
main = do
  s <- MWC.createSystemSeed
  printPPL $ program
  print $ evalState (stateHandler program) (s, mempty)
  where
    program = programOnPage226 [1] (+) (-) 2

