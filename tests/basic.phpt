--TEST--
Testa soma básica do ZMatrix
--SKIPIF--
<?php if (!extension_loaded("zmatrix")) die("skip"); ?>
--FILE--
<?php
$a = new ZMatrix\ZTensor([[1, 2], [3, 4]]);
$b = new ZMatrix\ZTensor([[5, 6], [7, 8]]);
$c = $a->add($b);
echo $c;
?>
--EXPECT--
[[6,8],
 [10,12]]
