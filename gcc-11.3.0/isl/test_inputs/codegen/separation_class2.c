{
  for (int c0 = 0; c0 < -(n % 8) + n; c0 += 8) {
    for (int c1 = 0; c1 < -(n % 8) + n; c1 += 8)
      for (int c2 = 0; c2 <= 7; c2 += 1)
        for (int c3 = 0; c3 <= 7; c3 += 1)
          A(c0 + c2, c1 + c3);
    for (int c2 = 0; c2 <= 7; c2 += 1)
      for (int c3 = 0; c3 < n % 8; c3 += 1)
        A(c0 + c2, -(n % 8) + n + c3);
  }
  for (int c1 = 0; c1 < n; c1 += 8)
    for (int c2 = 0; c2 < n % 8; c2 += 1)
      for (int c3 = 0; c3 <= min(7, n - c1 - 1); c3 += 1)
        A(-(n % 8) + n + c2, c1 + c3);
}
