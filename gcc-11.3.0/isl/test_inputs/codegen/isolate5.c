{
  for (int c0 = 0; c0 <= 9; c0 += 1) {
    if (c0 % 2 == 0) {
      for (int c1 = 0; c1 <= 1; c1 += 1)
        A(c0 / 2, c1);
    } else {
      for (int c1 = 0; c1 <= 1; c1 += 1)
        B((c0 - 1) / 2, c1);
    }
  }
  for (int c0 = 10; c0 <= 89; c0 += 1) {
    if (c0 % 2 == 0) {
      for (int c1 = 0; c1 <= 1; c1 += 1)
        A(c0 / 2, c1);
    } else {
      for (int c1 = 0; c1 <= 1; c1 += 1)
        B((c0 - 1) / 2, c1);
    }
  }
  for (int c0 = 90; c0 <= 199; c0 += 1) {
    if (c0 % 2 == 0) {
      for (int c1 = 0; c1 <= 1; c1 += 1)
        A(c0 / 2, c1);
    } else {
      for (int c1 = 0; c1 <= 1; c1 += 1)
        B((c0 - 1) / 2, c1);
    }
  }
}
