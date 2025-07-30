for (int c0 = 1; c0 <= 9; c0 += 1) {
  if (c0 >= 6) {
    for (int c1 = 1; c1 <= 9; c1 += 1)
      s0(c0, c1);
  } else if (c0 <= 4) {
    for (int c1 = 1; c1 <= 9; c1 += 1)
      s0(c0, c1);
  } else {
    for (int c1 = 1; c1 <= 9; c1 += 1) {
      s0(5, c1);
      s1(5, c1);
    }
  }
}
