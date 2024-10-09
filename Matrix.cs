using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace MLA
{
    public class Matrix
    {
        private int rows, cols;
        private double[,] data;
        private static ThreadLocal<Random> random = new ThreadLocal<Random>(() => new Random());
        private Size mySize;
        private delegate double Operation(double value, double operand);
        public Matrix(int rows, int cols)
        {
            this.rows = rows;
            this.cols = cols;
            this.mySize = new Size(rows, cols);
            this.data = new double[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                    this.data[r, c] = 0;
            }
        }
        public Matrix(double[] m)
        {
            this.rows = 1;
            this.cols = m.Length;
            this.mySize = new Size(rows, cols);
            this.data = new double[rows, cols];
            for (int c = 0; c < cols; c++)
                this.data[0, c] = m[c];
        }

        public Matrix(double[,] m)
        {
            this.rows = m.GetLength(0);
            this.cols = m.GetLength(1);
            this.mySize = new Size(rows, cols);
            this.data = new double[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                    this.data[r, c] = m[r, c];
            }
        }

        public Matrix(Matrix m)
        {
            this.rows = m.rows;
            this.cols = m.cols;
            this.mySize = new Size(rows, cols);
            this.data = new double[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0;c < cols; c++)
                {
                    this.data[r,c] = m.Grid[r, c];
                }
            }
        }

        public double[,] Grid { get { return this.data; } }

        private void Process(double operand, Operation operation)
        {
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                    this.data[r, c] = operation(this.data[r, c], operand);
            }
        }

        public void Transpose() {
            double[,] result = new double[cols, rows];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                    result[c, r] = this.data[r, c];
            }
            Replace(result);
        }

        public static Matrix Transpose(Matrix matrix)
        {
            Matrix result = (Matrix)matrix.MemberwiseClone();
            result.Transpose();
            return result;
        }

        public static Matrix fromArray(double[] array)
        {
            Matrix result = new Matrix(array.Length, 1);
            for (int i = 0; i < array.Length; i++)
            {
                result.Grid[i, 0] = array[i];
            }
            return result;
        }

        public static double[] toArray(Matrix m)
        {
            double[] result = new double[m.rows];
            for (int i = 0; i < result.Length; i++)
                result[i] = m.Grid[i, 0];
            return result;
        }

        private void Replace(double[,] result)
        {
            this.rows = result.GetLength(0);
            this.cols = result.GetLength(1);
            this.data = new double[rows, cols];
            for (int r = 0;r < this.rows; r++)
            {
                for (int c = 0;c< this.cols; c++)
                    this.data[r,c] = result[r,c];
            }
        }

        private double MultD(double value, double operand)
        {
            return value * operand;
        }

        public void SetAllToValue(double value)
        {
            Operation setValue = (double currentValue, double newValue) => newValue;
            Process(value, setValue);
        }

        private double AddD(double value, double operand)
        {
            return value + operand;
        }

        public void Mult(double m)
        {
            Process(m, MultD);
        }

        public void Sigmoid()
        {
            for (int r = 0; r < this.rows; r++)
            {
                for (int c = 0; c< this.cols; c++)
                    this.Grid[r,c] = Sig(this.Grid[r,c]);
            }
        }

        private double Sig(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }

        public void DSigmoid()
        {
            for (int r = 0; r < this.rows; r++)
            {
                for (int c = 0; c< this.cols; c++)
                {
                    double s = Sig(this.Grid[r,c]);
                    this.Grid[r, c] = s * (1 - s);
                }
            }
        } 

        public static Matrix Mult(Matrix m1, Matrix m2)
        {
            Matrix result = (Matrix)m1.MemberwiseClone();
            result.Mult(m2);
            return result;
        }

        public static Matrix Mult(Matrix m1, double multiplier)
        {
            Matrix result = (Matrix)m1.MemberwiseClone();
            result.Mult(multiplier);
            return result;
        }

        public void Mult(Matrix m)
        {
            if (m.GetSize() == this.mySize)
            {
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                        this.data[r, c] *= m.Grid[r, c];
                }
            }
        }

        public void Add(double add)
        {
            Process(add, AddD);
        }

        public void Add(Matrix m)
        {
            if (m.GetSize() == this.mySize)
            {
                for (int row = 0; row < rows; row++)
                {
                    for (int column = 0; column < cols; column++)
                    {
                        this.data[row, column] += m.Grid[row, column];
                    }
                }
            }
        }

        public static Matrix Add(Matrix m1, Matrix m2)
        {
            Matrix result = (Matrix)m1.MemberwiseClone(); 
            result.Add(m2);
            return result;
        }

        private double SubD(double value, double operand)
        {
            return value - operand;
        }

        public void Subtract(Matrix m)
        {
            if (m.GetSize() == this.mySize)
            {
                for (int row = 0; row < rows; row++)
                {
                    for (int column = 0; column < cols; column++)
                    {
                        this.data[row, column] -= m.Grid[row, column];
                    }
                }
            }
        }

        public static Matrix Subtract(Matrix m1, Matrix m2)
        {
            Matrix result = (Matrix)m1.MemberwiseClone();
            result.Subtract(m2);
            return result;
        }

        public void Subtract(double value)
        {
            Process(value, SubD);
        }

        public Size GetSize()
        {
            return this.mySize;
        }

        public int Rows
        {
            get { return rows; }
        }

        public int Columns
        {
            get { return cols; }
        }

        public void Randomize()
        {
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                    this.data[row, col] = Math.Floor(random.Value.NextDouble() * 10);
            }
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            for (int row = 0; row < rows; row++)
            {
                sb.Append("[");
                for (int col = 0; col < cols; col++)
                {
                    sb.Append(this.data[row, col].ToString());
                    sb.Append(", ");
                }
                sb.Append("]\n");
            }
            return sb.ToString();
        }

        public static Matrix Dot(Matrix m1, Matrix m2)
        {

            if ((m1.Grid.GetLength(1) == m2.Grid.GetLength(0)))
            {
                Matrix final = new Matrix(m1.rows, m2.cols);
                Matrix hmat, vmat;
                for (int r = 0; r < m1.rows; r++)
                {
                    for (int c = 0; c < m2.cols; c++)
                    {
                        hmat = m1.GetRow(r);
                        vmat = m2.ColAsRow(c);
                        hmat.Mult(vmat);
                        final.Grid[r, c] = hmat.GetRowSum(0);
                    }
                }
                return final;
            }
            return null;
        }
        public void Dot(Matrix m)
        {

            if ((this.Grid.GetLength(1) == m.Grid.GetLength(0)))
            {
                double[,] final = new double[this.rows, m.Grid.GetLength(1)];
                Matrix hmat, vmat;
                for (int r = 0; r < this.rows; r++)
                {
                    for (int c = 0; c < m.cols; c++)
                    {
                        hmat = this.GetRow(r);
                        vmat = m.ColAsRow(c);
                        hmat.Mult(vmat);
                        final[r, c] = hmat.GetRowSum(0);
                    }
                }
                this.data = null;
                this.rows = final.GetLength(0);
                this.cols = final.GetLength(1);
                data = new double[rows, cols]; 
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                        data[r,c] = final[r,c];
                }
                final = null;
            }
        }

        public Matrix ColAsRow(int columnNumber)
        {
            if (this.data.GetLength(1) > columnNumber)
            {
                Matrix final = new Matrix(1, this.rows);
                for (int row = 0; row < rows; row++)
                {
                    final.Grid[0, row] = this.data[row, columnNumber];
                }
                return final;
            }
            return null;
        }

        public Matrix GetRow(int rowNumber)
        {
            Matrix final = new Matrix(1, this.cols);
            for (int col = 0; col < this.cols; col++)
                final.Grid[0, col] = this.data[rowNumber, col];
            return final;
        }

        public double GetRowSum(int RowNumber)
        {
            double sum = 0;
            for (int col = 0; col < this.cols; col++)
                sum += this.data[RowNumber, col];
            return sum;
        }

        internal double[] toArray()
        {
            double[] result = new double[this.rows];
            for (int i = 0; i < this.rows; i++)
                result[i] = this.data[i,0];
            return result;
            
        }

        internal Matrix Clone()
        {
            Matrix result = new Matrix(this.rows, this.cols);
            for (int r = 0; r < this.rows; r++)
            {
                for (int c = 0; c < this.cols; c++)
                {
                    result.Grid[r, c] = this.data[r, c];
                }
            }
            return result;
        }
    }
}
