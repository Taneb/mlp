{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE QuantifiedConstraints #-}
module MLP (module MLP.Core, module MLP.Classes, module MLP.Layers, module MLP) where

import Data.List (foldl')
import GHC.TypeNats
import Linear.V

import MLP.Core
import MLP.Classes
import MLP.Layers

-- training
--
-- List of (input, output) pairs
-- final cost function: quadratic difference
-- hooking it all together


train 
  :: (forall l. c l => (BackPropagateCPU l, NormalFormNetwork l)
     ,KnownNat o
     )
  => [(V i Double, V o Double)]
  -> Double
  -> Network c ('Tip '[i]) ('Tip '[o])
  -> Network c ('Tip '[i]) ('Tip '[o])
train dat learningRate n = foldl' trainExample' n dat
  where
    trainExample net (input, ref) = 
      let ((output, _), net') = backPropagateCPU learningRate net (input, cost)
          cost = output - ref
      in net'

    trainExample' net ir =
      let net' = trainExample net ir
      in normalFormNetwork net' `seq` net'
