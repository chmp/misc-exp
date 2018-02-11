#include <stdexcept>
#include <string>
#include <tuple>
#include <iostream>
#include <cstdint>
#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

namespace hexworld {

// inspired by https://www.redblobgames.com/grids/hexagons/

// tag class for internal use
struct internal {};

// NOTE: use axial coordinates internally: q = x, r = z, y = -q - r
struct CubePoint {
    const int q = 0;
    const int r = 0;

    CubePoint() {}
    CubePoint(int q, int r, internal) : q(q), r(r) {}
    CubePoint(int x, int y, int z) : q(x), r(z) {
        if((x + y + z) != 0) {
            throw std::domain_error("components have to sum to one");
        }
    }

    int x() const { return q; }
    int y() const { return -q - r; }
    int z() const { return r; }

    bool operator==(CubePoint b) const { return (q == b.q) && (r == b.r); }
    bool operator!=(CubePoint b) const { return !(*this == b); }
    CubePoint operator+(CubePoint b) const { return {q + b.q, r + b.r, internal{}}; }
    CubePoint operator-(CubePoint b) const { return {q - b.q, r - b.r, internal{}}; }
};

// use odd-q format
struct OffsetPoint {
    const int col = 0;
    const int row = 0;

    OffsetPoint() {}
    OffsetPoint(int col, int row) : col(col), row(row) {}

    bool operator==(OffsetPoint b) const { return (col == b.col) && (row == b.row); }
};

OffsetPoint to_offset(OffsetPoint p) {
    return p;
}

OffsetPoint to_offset(CubePoint p) {
    return {p.x(), p.z() + (p.x() - (p.x() & 1)) / 2};
}

CubePoint to_cube(CubePoint p) {
    return p;
}

CubePoint to_cube(OffsetPoint p) {
    int x = p.col;
    int z = p.row - (p.col - (p.col & 1)) / 2;
    return {x, -x - z, z};
}

CubePoint rotate(CubePoint p, int rotation) {
    // +60: {x, y, z} -> {-z, -x, -y}
    // -60: {x, y, z} -> {-y, -z, -x}
    if(rotation < 0) {
        // NOTE: integer casting trucates the zeros, i.e., -1.5 -> -1.0 add +1 to compensate
        rotation = rotation + 6 * (1 + (rotation / 6));
        rotation = rotation % 6;
    }
    
    switch(rotation) {
        case 0: return {+p.x(), +p.y(), +p.z()};
        case 1: return {-p.z(), -p.x(), -p.y()};
        case 2: return {+p.y(), +p.z(), +p.x()};
        case 3: return {-p.x(), -p.y(), -p.z()};
        case 4: return {+p.z(), +p.x(), +p.y()};
        case 5: return {-p.y(), -p.z(), -p.x()};
        default: throw std::domain_error("can not handle rotation: " + std::to_string(rotation));
    }
}

OffsetPoint rotate(OffsetPoint p, int rotation) {
    return to_offset(rotate(to_cube(p), rotation));
}

template<typename T>
struct Buffer {
    const int width = 0;
    const int height = 0;
    std::vector<T> data;

    Buffer() {}
    Buffer(int width, int height) : width(width), height(height), data(width * height, T()) {}

    ~Buffer() {
        std::cout << "Deleted" << std::endl;
    }

    T get_(int row, int col) const { 
        return data[width * row + col]; 
    }
    
    void set_(int row, int col, T value) {
        data[width * row + col] = value;
    }

    bool inside(OffsetPoint p) const {
        return (p.col >= 0) && (p.col < width) && (p.row >= 0) && (p.row < height);
    }

    T get(OffsetPoint p) const {
        if(!inside(p)) {
            return T();
        }
        return get_(p.row, p.col);
    }
    void set(OffsetPoint p, T value) {
        if(!inside(p)) {
            // TODO: throw?
            return;
        }
        set_(p.row, p.col, value);
    }
    
    bool inside(CubePoint p) const { return inside(to_offset(p)); }
    T get(CubePoint p) const {return get(to_offset(p)); }
    void set(CubePoint p, T value) { set(to_offset(p), value); }

    bool operator==(const Buffer<T>& other) const {
        return (width == other.width) && (height == other.height) && (data == other.data);
    }
};

}

namespace hexworld::detail {

namespace py = pybind11;

template<typename T, typename U>
constexpr bool fits_range(U value) {
    return (std::numeric_limits<T>::lowest() <= value) && (value <= std::numeric_limits<T>::max());
}

template<typename T>
void create_buffer_bindings(const char* name, py::module m) {

    py::class_<Buffer<T>>(m, name, py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def_buffer([](Buffer<T> &b) -> py::buffer_info {
            return py::buffer_info(
                b.data.data(), sizeof(T), py::format_descriptor<T>::format(),
                // ndim, shape, strides
                2, { b.height, b.width }, { sizeof(T) * b.width, sizeof(T)}  
            );
        })
        .def_readonly("width", &Buffer<T>::width)
        .def_readonly("height", &Buffer<T>::height)
        .def("inside", [](Buffer<T>& b, int row, int col) -> bool {
            return b.inside(OffsetPoint{row, col});
        })
        .def(py::self == py::self)
        .def("inside", py::overload_cast<OffsetPoint>(&Buffer<T>::inside, py::const_))
        .def("inside", py::overload_cast<CubePoint>(&Buffer<T>::inside, py::const_))
        .def("__getitem__", [](const Buffer<T>& b, std::tuple<int, int> p) -> T {
            return b.get(OffsetPoint{std::get<0>(p), std::get<1>(p)});
        })
        .def("__getitem__", py::overload_cast<OffsetPoint>(&Buffer<T>::get, py::const_))
        .def("__getitem__", py::overload_cast<CubePoint>(&Buffer<T>::get, py::const_))
        .def("__setitem__", [](Buffer<T>& b, std::tuple<int, int> p, T value) {
            b.set(OffsetPoint{std::get<0>(p), std::get<1>(p)}, value);
        })
        .def("__setitem__", py::overload_cast<OffsetPoint, T>(&Buffer<T>::set))
        .def("__setitem__", py::overload_cast<CubePoint, T>(&Buffer<T>::set));

    m.def(("_to_" + std::string(name)).c_str(), [](py::buffer b) -> Buffer<T> {
        py::buffer_info info = b.request();
        auto expected_format = py::format_descriptor<T>::format();

        /*if (info.format != expected_format) {
            throw std::runtime_error("Incompatible type: " + info.format + " != " + expected_format);
        }*/
        if (info.ndim != 2) {
            throw std::runtime_error("Incompatible buffer dimension: " + std::to_string(info.ndim));
        }
        if(!fits_range<int>(info.shape[0]) || !fits_range<int>(info.shape[1])) {
            throw std::runtime_error("Shape too large");
        }

        auto sy = info.strides[0], sx = info.strides[1];
        int h = info.shape[0], w = info.shape[1];
        auto data = static_cast<char*>(info.ptr);

        Buffer<T> result{w, h};
        for(int y = 0; y < h; ++y) {
            for(int x = 0; x < w; ++x) {
                T value = *reinterpret_cast<T*>(data + x * sx + y * sy);
                result.set_(y, x, value);
            }
        }
        return result;
    });
}

}

PYBIND11_MODULE(_hexworld, m) {
    using namespace hexworld;
    namespace py = pybind11;

    m.doc() = "Support for hexworld simulations";
    
    py::class_<CubePoint>(m, "CubePoint")
        .def(py::init<>())
        .def(py::init<int, int, int>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_property_readonly("x", &CubePoint::x)
        .def_property_readonly("y", &CubePoint::y)
        .def_property_readonly("z", &CubePoint::z)
        .def(py::self == py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("__str__", [](CubePoint p) -> std::string {
            return "hexworld.CubePoint("
                "x=" + std::to_string(p.x()) + ", " + 
                "y=" + std::to_string(p.y()) + ", " +
                "z=" + std::to_string(p.z()) + ")";
        });

    py::class_<OffsetPoint>(m, "OffsetPoint")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("col"), py::arg("row"))
        .def_readonly("col", &OffsetPoint::col)
        .def_readonly("row", &OffsetPoint::row)
        .def("__str__", [](OffsetPoint p) -> std::string {
            return "hexworld.OffsetPoint("
                "col=" + std::to_string(p.col) + ", " + 
                "row=" + std::to_string(p.row) + ")";
        });
    
    hexworld::detail::create_buffer_bindings<std::int32_t>("Int32Buffer", m);
    hexworld::detail::create_buffer_bindings<std::int64_t>("Int64Buffer", m);
    
    m.def("to_offset", py::overload_cast<OffsetPoint>(&to_offset));
    m.def("to_offset", py::overload_cast<CubePoint>(&to_offset));

    m.def("to_cube", py::overload_cast<OffsetPoint>(&to_cube));
    m.def("to_cube", py::overload_cast<CubePoint>(&to_cube));

    m.def("rotate", py::overload_cast<CubePoint, int>(&rotate));
    m.def("rotate", py::overload_cast<OffsetPoint, int>(&rotate));
}
