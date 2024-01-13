export class ComplexNumber {
    constructor(real, imaginary) {
        this.real = real;
        this.imaginary = imaginary;
    }

    add(cn) {
        return new ComplexNumber(
            this.real + cn.real,
            this.imaginary + cn.imaginary
        )
    }

    mult(cn) {
        return new ComplexNumber(
            (this.real * cn.real) - (this.imaginary * cn.imaginary),
            (this.real * cn.imaginary) + (this.imaginary * cn.real)
        )
    }

    square() {
        return this.mult(this)
    }

    iterate(orig) {
        return this.square().add(orig);
    }

    toString() {
        return this.real + " + " + this.imaginary + "i";
    }
}

export class MapRange {
    constructor(xRange, yRange, minReal, maxReal, minImaginary, maxImaginary) {
        this.xRange = xRange;
        this.yRange = yRange;
        this.minReal = minReal;
        this.minImaginary = minImaginary;
        this.maxReal = maxReal;
        this.maxImaginary = maxImaginary;
        this.realinc = Math.abs(this.maxReal - this.minReal) / this.xRange;
        this.imaginc = Math.abs(this.maxImaginary - this.minImaginary) / this.yRange;    
    }

    pixelToComplex(x, y) {
        return new ComplexNumber(this.minReal + x*this.realinc, this.maxImaginary - y*this.imaginc)
    }

    complexToPixel(cn) {
        return [this.xRange * (cn.real - this.minReal) / (this.maxReal - this.minReal), this.yRange * (this.maxImaginary - cn.imaginary) / (this.maxImaginary - this.minImaginary)];
    }

}
