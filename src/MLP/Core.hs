{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
module MLP.Core where

import Control.Category
import GHC.Exts (Constraint)
import GHC.TypeNats
import Linear.V

import Prelude hiding ((.), id)

-- | n-dimensional tensor with dimensions encoded in the type. Implemented as
-- nested 'V's. First dimension listed is outermost.
type family Tensor (dims :: [Nat]) :: * where
  Tensor '[] = Double
  Tensor (x ': xs) = V x (Tensor xs)

-- | Binary tree with data in the nodes.
data BinTree a = Tip a | Fork (BinTree a) (BinTree a)

-- | Given a binary tree of dimensions, a type consisting of nested tuples with
-- 'Tensor's of the corresponding dimensions.
type family TensorTree (dims :: BinTree [Nat]) :: * where
  TensorTree (Tip xs) = Tensor xs
  TensorTree (Fork l r) = (TensorTree l, TensorTree r)

infixr 1 `Cons`

-- | Given a constraint @c@ on layers of kind @'BinTree' ['Nat'] -> 'BinTree' ['Nat'] -> *@,
-- a free category, or type-indexed list of these layers.
data Network :: ((BinTree [Nat] -> BinTree [Nat] -> *) -> Constraint) -> BinTree [Nat] -> BinTree [Nat] -> * where
  Nil :: Network c x x
  Cons :: c l => l i m -> Network c m o -> Network c i o

instance Category (Network c) where
  id = Nil
  x . Nil = x
  x . Cons l y = Cons l (x . y)

-- | Embed a layer into an appropriate network.
forget :: c l => l i o -> Network c i o
forget l = Cons l Nil

-- | Given a transformation from layers to some 'Category', transform a 'Network'
-- into that 'Category'.
--
-- @
--   'free' f . 'forget' = f
-- @
free :: Category k => (forall l i o. c l => l i o -> k i o) -> Network c i o -> k i o
free f Nil = id
free f (Cons l n) = free f n . f l
