domain: "[m, n] -> { S_14[j] : 0 <= j < m; S_19[i, j] : 0 <= i < n and 0 <= j < m; S_27[i, j, k] : i >= 0 and i < j < m and 0 <= k < n; S_29[i, j] : i >= 0 and i < j < m }"
child:
  context: "[m, n] -> { [] : 0 < m <= 2147483647 and -2147483648 <= n <= 2147483647 }"
  child:
            schedule: "[m, n] -> [{ S_19[i, j] -> [(0)]; S_29[i, j] -> [(32*floor((j)/32))]; S_27[i, j, k] -> [(32*floor((i)/32))]; S_14[j] -> [(0)] }, { S_19[i, j] -> [(32*floor((i)/32))]; S_29[i, j] -> [(32*floor((n)/32))]; S_27[i, j, k] -> [(32*floor((k)/32))]; S_14[j] -> [(0)] }, { S_19[i, j] -> [(32*floor((j)/32))]; S_29[i, j] -> [(32*floor((i + j)/32))]; S_27[i, j, k] -> [(32*floor((j)/32))]; S_14[j] -> [(32*floor((j)/32))] }]"
            permutable: 1
            coincident: [ 1, 1, 1 ]
            options: "{ atomic[i0] : 0 <= i0 <= 2 }"
            child:
              schedule: "[m, n] -> [{ S_19[i, j] -> [(0)]; S_29[i, j] -> [(j - 32*floor((j)/32))]; S_27[i, j, k] -> [(i - 32*floor((i)/32))]; S_14[j] -> [(0)] }, { S_19[i, j] -> [(i - 32*floor((i)/32))]; S_29[i, j] -> [(n - 32*floor((n)/32))]; S_27[i, j, k] -> [(k - 32*floor((k)/32))]; S_14[j] -> [(0)] }, { S_19[i, j] -> [(j - 32*floor((j)/32))]; S_29[i, j] -> [(i + j - 32*floor((i + j)/32))]; S_27[i, j, k] -> [(j - 32*floor((j)/32))]; S_14[j] -> [(j - 32*floor((j)/32))] }]"
              permutable: 1
              coincident: [ 1, 1, 1 ]
              child:
                sequence:
                - filter: "[m, n] -> { S_29[i, j] }"
                - filter: "[m, n] -> { S_14[j] }"
                - filter: "[m, n] -> { S_19[i, j] }"
                - filter: "[m, n] -> { S_27[i, j, k] }"
