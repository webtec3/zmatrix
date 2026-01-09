#!/usr/bin/env bash
set -euo pipefail

# Métodos a testar
methods=(add sub mul divide greater)

# Função auxiliar para executar um fragmento PHP isolado
# $1 = descrição, $2 = PHP code
run_php() {
  desc="$1"
  php -ddisplay_errors=stderr -r "$2" 2>&1 | sed "s|^|[$desc] |"
  status=$?
  if [ $status -eq 139 ]; then
    echo "[$desc] → SEGMENTATION FAULT"
  elif [ $status -ne 0 ]; then
    echo "[$desc] → ERROR (exit code $status)"
  fi
}

echo "=== Iniciando testes ZTensor ops ==="

for m in "${methods[@]}"; do
  echo
  echo "---- Método $m ----"

  # 1) Escalar puro
  run_php "$m-scalar" "
    \$t = ZMatrix\\ZTensor::arr([1,2,3]);
    \$t->$m(2.5);
    echo 'ok: '.json_encode(\$t->toArray());
  "

  # 2) Vetor 1D de tamanho 1 (deve tratar como escalar)
  run_php "$m-vec1" "
    \$t = ZMatrix\\ZTensor::arr([1,2,3]);
    \$t->$m([2.5]);
    echo 'ok: '.json_encode(\$t->toArray());
  "

  # 3) Mesma forma
  run_php "$m-same-shape" "
    \$t = ZMatrix\\ZTensor::arr([[1,2],[3,4]]);
    \$t->$m([[10,20],[30,40]]);
    echo 'ok: '.json_encode(\$t->toArray());
  "

  # 4) Broadcast 2D×1D
  run_php "$m-broadcast" "
    \$t = ZMatrix\\ZTensor::arr([[1,2],[3,4]]);
    \$t->$m([100,200]);
    echo 'ok: '.json_encode(\$t->toArray());
  "

  # 5) Broadcast inverso (deve lançar exceção)
  run_php "$m-inv-bc" "
    try {
      \$t = ZMatrix\\ZTensor::arr([100,200]);
      \$t->$m([[1,2],[3,4]]);
      echo 'should not get here';
    } catch (Exception \$e) {
      echo 'caught: '.\$e->getMessage();
    }
  "
done

echo
echo "=== Fim dos testes ==="
